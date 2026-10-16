"""Append-only, hash-chained claim ledger in JSONL.

Every record carries ``prev`` (hash of the previous record, or ``GENESIS``)
and ``hash`` (sha256 over the record's canonical material). A separate
head file records the last hash and sequence number, so ``verify_chain``
can detect truncation as well as edits: a chain that was shortened still
has valid internal links, but it no longer ends at the recorded head.

The point of logging a claim *at the moment it is made*, together with the
check that would settle it and the value observed right then, is that the
denominator cannot be chosen after the verdicts are known.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

GENESIS = "GENESIS"
HASH_LEN = 64
CheckSpec = Union[str, Callable[[], Any]]

__all__ = [
    "Claim",
    "ChainError",
    "ChainReport",
    "Ledger",
    "canonical_material",
    "hash_record",
    "verify_chain",
]


class ChainError(ValueError):
    """Raised when a chain fails verification and the caller asked to raise."""


@dataclass(frozen=True)
class Claim:
    """One ledger record.

    ``check`` is a shell command (string) or the dotted name of a callable;
    callables are recorded by name, never pickled. ``observed`` is whatever the
    check produced at claim time, stringified. ``kind`` is a free label such as
    ``numeric``, ``existence``, ``state``, ``causal``, ``prediction``.
    """

    seq: int
    ts: str
    claim: str
    check: str
    observed: str
    kind: str
    prev: str
    hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        return cls(
            seq=int(d["seq"]),
            ts=str(d["ts"]),
            claim=str(d["claim"]),
            check=str(d["check"]),
            observed=str(d["observed"]),
            kind=str(d["kind"]),
            prev=str(d["prev"]),
            hash=str(d.get("hash", "")),
            meta=dict(d.get("meta") or {}),
        )


def canonical_material(rec: Claim) -> str:
    """The exact bytes that are hashed. Every field except ``hash`` itself.

    ``meta`` is included via a sorted-key JSON dump so that annotations are
    tamper-evident too. Changing this function invalidates every existing chain,
    which is the intended property.
    """
    meta = json.dumps(rec.meta, sort_keys=True, ensure_ascii=False)
    return "\x1f".join(
        [str(rec.seq), rec.ts, rec.claim, rec.check, rec.observed, rec.kind, meta, rec.prev]
    )


def hash_record(rec: Claim) -> str:
    return hashlib.sha256(canonical_material(rec).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    rows: int
    head: str
    reason: str
    at_seq: Optional[int] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _check_name(check: CheckSpec) -> str:
    if isinstance(check, str):
        return check
    mod = getattr(check, "__module__", "") or ""
    qual = getattr(check, "__qualname__", None) or getattr(check, "__name__", repr(check))
    return f"callable:{mod}.{qual}" if mod else f"callable:{qual}"


class Ledger:
    """A JSONL ledger with a sidecar head file (``<path>.head``).

    Records are only ever appended. The head file stores ``{"seq": n, "hash": h}``
    for the last record written; ``verify_chain`` requires the file to end there.
    """

    def __init__(self, path: Union[str, os.PathLike[str]]):
        self.path = Path(path)
        self.head_path = Path(str(self.path) + ".head")

    # ---- reading ---------------------------------------------------------
    def records(self) -> list[Claim]:
        if not self.path.exists():
            return []
        out: list[Claim] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Claim.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ChainError(f"line {line_no}: unreadable record ({exc})") from exc
        return out

    def __iter__(self) -> Iterator[Claim]:
        return iter(self.records())

    def __len__(self) -> int:
        return len(self.records())

    def read_head(self) -> Optional[dict[str, Any]]:
        if not self.head_path.exists():
            return None
        return json.loads(self.head_path.read_text(encoding="utf-8"))

    # ---- writing ---------------------------------------------------------
    def append(
        self,
        claim: str,
        check: CheckSpec,
        observed: Any,
        kind: str,
        *,
        ts: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> Claim:
        """Append one record. The chain is verified before the write so a
        corrupted file is never extended."""
        report = verify_chain(self)
        if not report.ok and report.rows > 0:
            raise ChainError(f"refusing to append to a broken chain: {report.reason}")
        rows = self.records()
        prev = rows[-1].hash if rows else GENESIS
        rec = Claim(
            seq=(rows[-1].seq + 1) if rows else 1,
            ts=ts or _now_iso(),
            claim=claim,
            check=_check_name(check),
            observed="" if observed is None else str(observed),
            kind=kind,
            prev=prev,
            meta=dict(meta or {}),
        )
        rec = Claim(**{**asdict(rec), "hash": hash_record(rec)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(rec.to_json() + "\n")
        self._write_head(rec)
        return rec

    def _write_head(self, rec: Claim) -> None:
        tmp = self.head_path.with_suffix(self.head_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"seq": rec.seq, "hash": rec.hash}), encoding="utf-8")
        os.replace(tmp, self.head_path)


def verify_chain(ledger: Union[Ledger, str, os.PathLike[str]], *, raise_on_fail: bool = False) -> ChainReport:
    """Verify prev-links, recomputed hashes, and that the file ends at the head.

    Detects: edited fields (hash mismatch), re-ordered or spliced rows (prev
    mismatch, seq gap), truncation (last hash != recorded head), and a head
    file pointing past the data (same check). An empty ledger with no head is OK.
    """
    led = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
    try:
        rows = led.records()
    except ChainError as exc:
        rep = ChainReport(False, 0, "", f"unreadable: {exc}")
        if raise_on_fail:
            raise ChainError(rep.reason) from exc
        return rep
    head = led.read_head()

    def fail(reason: str, at: Optional[int]) -> ChainReport:
        rep = ChainReport(False, len(rows), rows[-1].hash if rows else "", reason, at)
        if raise_on_fail:
            raise ChainError(reason)
        return rep

    prev = GENESIS
    expected_seq = 1
    for r in rows:
        if r.seq != expected_seq:
            return fail(f"seq gap: expected {expected_seq}, found {r.seq}", r.seq)
        if r.prev != prev:
            return fail("prev mismatch (row spliced, reordered, or an earlier row removed)", r.seq)
        if len(r.hash) != HASH_LEN or hash_record(r) != r.hash:
            return fail("hash mismatch (record edited after writing)", r.seq)
        prev = r.hash
        expected_seq += 1

    if head is None:
        if rows:
            return fail("head file missing: cannot prove the chain was not truncated", rows[-1].seq)
        return ChainReport(True, 0, "", "empty ledger")
    if not rows:
        return fail("head recorded but ledger is empty (truncated to zero)", None)
    if head.get("hash") != rows[-1].hash or int(head.get("seq", -1)) != rows[-1].seq:
        return fail(
            f"chain ends at seq {rows[-1].seq} but head records seq {head.get('seq')} (truncated or head stale)",
            rows[-1].seq,
        )
    return ChainReport(True, len(rows), rows[-1].hash, "ok")
