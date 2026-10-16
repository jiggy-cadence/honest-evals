"""Planted-item canaries: a checker may not report "clean" until it has
caught a known-false item and passed a known-true one.

The checker is any callable ``checker(data) -> flagged`` where ``flagged`` is
an iterable of the items (or item keys) it considers wrong. The canary copies
``data``, appends a known-false and a known-true item, runs the checker on the
copy, and only then lets the checker's verdict on the *real* data through.

A checker that flags nothing fails (it missed the planted false item). A
checker that flags everything fails (it flagged the planted true item). Both
failures raise ``CanaryFailed`` instead of returning anything, so "clean" can
never be reported by accident.

Use as a decorator::

    @canary(false_item={"seq": -1, "strict": True}, true_item={"seq": -2}, key=lambda r: r["seq"])
    def find_errors(rows): ...

or as a context manager::

    with Canary(rows, false_item=bad, true_item=good, key=key) as c:
        c.observe(find_errors(c.data))
    real_flags = find_errors(rows)
"""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import Any, Callable, Generic, Hashable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
Flagged = Iterable[Any]
Checker = Callable[[Sequence[T]], Flagged]

__all__ = ["Canary", "CanaryFailed", "CanaryVerdict", "canary", "run_canary"]


class CanaryFailed(AssertionError):
    """The checker cannot be trusted to say 'clean'."""

    def __init__(self, verdict: "CanaryVerdict"):
        super().__init__(verdict.reason)
        self.verdict = verdict


@dataclass(frozen=True)
class CanaryVerdict:
    passed: bool
    caught_false: bool
    passed_true: bool
    reason: str
    flagged_on_copy: int


def _identity(x: Any) -> Any:
    return x


def run_canary(
    checker: Checker[T],
    data: Sequence[T],
    *,
    false_item: T,
    true_item: T,
    key: Callable[[T], Hashable] = _identity,
) -> CanaryVerdict:
    """Run ``checker`` on a copy of ``data`` with both plants appended."""
    planted: List[T] = list(copy.deepcopy(data))
    planted.append(copy.deepcopy(false_item))
    planted.append(copy.deepcopy(true_item))
    try:
        flagged = checker(planted)
    except Exception as exc:  # noqa: BLE001 - a crashing checker is not a passing one
        return CanaryVerdict(False, False, False, f"checker raised {exc.__class__.__name__}: {exc}", 0)
    # The checker may return items or already-extracted keys; accept both.
    flagged_keys: set[Hashable] = set()
    for f in flagged:
        try:
            flagged_keys.add(key(f))
        except (TypeError, KeyError, AttributeError, IndexError):
            flagged_keys.add(f)
    fk, tk = key(false_item), key(true_item)
    caught = fk in flagged_keys
    passed_true = tk not in flagged_keys
    if caught and passed_true:
        reason = "canary PASS: planted false item caught, planted true item passed"
    elif not caught and not passed_true:
        reason = "canary FAIL: checker is inverted or random (missed the false item, flagged the true one)"
    elif not caught:
        reason = "canary FAIL: planted false item was not caught (a no-op checker would look clean)"
    else:
        reason = "canary FAIL: planted true item was flagged (checker flags indiscriminately)"
    return CanaryVerdict(caught and passed_true, caught, passed_true, reason, len(flagged_keys))


def canary(
    *,
    false_item: Any,
    true_item: Any,
    key: Callable[[Any], Hashable] = _identity,
) -> Callable[[Checker[T]], Callable[[Sequence[T]], List[Any]]]:
    """Decorator: the wrapped checker runs its canary before every real call."""

    def deco(checker: Checker[T]) -> Callable[[Sequence[T]], List[Any]]:
        @functools.wraps(checker)
        def guarded(data: Sequence[T]) -> List[Any]:
            verdict = run_canary(checker, data, false_item=false_item, true_item=true_item, key=key)
            if not verdict.passed:
                raise CanaryFailed(verdict)
            return list(checker(data))

        guarded.canary_verdict = lambda data: run_canary(checker, data, false_item=false_item, true_item=true_item, key=key)  # type: ignore[attr-defined]
        return guarded

    return deco


class Canary(Generic[T]):
    """Context manager. Exposes ``.data`` (the planted copy); the body must call
    ``observe(flagged)`` with what the checker flagged on it. Leaving the block
    without a passing observation raises ``CanaryFailed``."""

    def __init__(
        self,
        data: Sequence[T],
        *,
        false_item: T,
        true_item: T,
        key: Callable[[T], Hashable] = _identity,
    ):
        self._key = key
        self.false_item = copy.deepcopy(false_item)
        self.true_item = copy.deepcopy(true_item)
        self.data: List[T] = list(copy.deepcopy(data)) + [self.false_item, self.true_item]
        self.verdict: Optional[CanaryVerdict] = None

    def observe(self, flagged: Flagged) -> CanaryVerdict:
        self.verdict = run_canary(lambda _d: flagged, self.data, false_item=self.false_item, true_item=self.true_item, key=self._key)
        return self.verdict

    def __enter__(self) -> "Canary[T]":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            return None
        if self.verdict is None:
            raise CanaryFailed(CanaryVerdict(False, False, False, "canary FAIL: checker never reported what it flagged", 0))
        if not self.verdict.passed:
            raise CanaryFailed(self.verdict)
        return None
