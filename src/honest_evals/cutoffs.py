"""Threshold sweeps: is the finding stable under the cutoff you chose?

Any analysis with a tunable number (a p-value cutoff, a drift window, an
error definition, a minimum-n) must show that its conclusion does not flip
when that number moves. ``sweep`` evaluates the analysis at >= 3 settings
and returns either a ``StableFinding`` (same winner everywhere) or a
``FragileFinding`` (the winner changed), never a bare result.

``evaluate(setting)`` may return:

* a verdict - any hashable value (``"A"``, ``True``, ``("ledger", "higher")``)
* a ranking - a list/tuple whose first element is the winner

The sweep compares winners, and for rankings also records whether the whole
order was preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Sequence, Union

__all__ = ["FragileFinding", "StableFinding", "SweepPoint", "sweep"]

MIN_SETTINGS = 3


@dataclass(frozen=True)
class SweepPoint:
    setting: Any
    winner: Hashable
    ranking: tuple[Hashable, ...]
    verdict: Any


@dataclass(frozen=True)
class StableFinding:
    winner: Hashable
    points: tuple[SweepPoint, ...]
    order_stable: bool  # full ranking identical at every setting (True for bare verdicts)

    @property
    def fragile(self) -> bool:
        return False

    def summary(self) -> str:
        return (f"STABLE: winner {self.winner!r} at all {len(self.points)} settings "
                f"({', '.join(repr(p.setting) for p in self.points)})"
                + ("" if self.order_stable else "; full ranking order changed below the winner"))


@dataclass(frozen=True)
class FragileFinding:
    points: tuple[SweepPoint, ...]
    winners: dict[Hashable, list[Any]] = field(default_factory=dict)  # winner -> settings where it won

    @property
    def fragile(self) -> bool:
        return True

    @property
    def winner(self) -> None:
        return None  # a fragile finding has no winner to report

    def summary(self) -> str:
        parts = [f"{w!r} wins at {s}" for w, s in self.winners.items()]
        return f"FRAGILE: winner depends on the threshold - " + "; ".join(parts)


def _as_ranking(value: Any) -> tuple[Hashable, ...]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("evaluate returned an empty ranking")
        return tuple(value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("evaluate must return a hashable verdict or a list/tuple ranking") from exc
    return (value,)


def sweep(
    settings: Sequence[Any],
    evaluate: Callable[[Any], Any],
) -> Union[StableFinding, FragileFinding]:
    """Evaluate at every setting; refuse fewer than three."""
    if len(settings) < MIN_SETTINGS:
        raise ValueError(f"a sweep needs at least {MIN_SETTINGS} settings, got {len(settings)}")
    if len(set(map(repr, settings))) != len(settings):
        raise ValueError("settings must be distinct")
    points: list[SweepPoint] = []
    for s in settings:
        verdict = evaluate(s)
        ranking = _as_ranking(verdict)
        points.append(SweepPoint(s, ranking[0], ranking, verdict))
    winners: dict[Hashable, list[Any]] = {}
    for p in points:
        winners.setdefault(p.winner, []).append(p.setting)
    if len(winners) > 1:
        return FragileFinding(tuple(points), winners)
    order_stable = len({p.ranking for p in points}) == 1
    return StableFinding(points[0].winner, tuple(points), order_stable)
