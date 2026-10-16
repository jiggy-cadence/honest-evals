"""Re-execute a claim's read-only check and classify the outcome.

Classes (``RerunClass``):

* ``still_true``       - the check reproduces the observed value today.
* ``drifted``          - the value differs because the data legitimately
                         changed since claim time (append-only logs, live
                         boards). Never an error at publish. Requires an
                         adjudication or a ``snapshot`` flag on the record.
* ``wrong_at_publish`` - the claim disagreed with its own check *when it was
                         made*. Established either by an explicit adjudication
                         or by record-consistency: the record carries the value
                         the claim asserted (``expected``) and it differs from
                         the value the check produced at the time (``observed``).
* ``not_rerunnable``   - refused (denylist, no command, timeout, crash) or a
                         mismatch nobody has adjudicated. Always carries a
                         named reason. These rows count as *unresolved* in the
                         denominator, never as true.

A mismatch is deliberately never auto-classified as drift: that would turn the
drift bucket into a place to hide errors. Unadjudicated mismatches are returned
as ``not_rerunnable`` with reason ``mismatch_unadjudicated`` so they stay
visible as unresolved.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Union

__all__ = [
    "Adjudication",
    "DENYLIST",
    "RerunClass",
    "RerunResult",
    "refusal_reason",
    "rerun",
    "rerun_many",
]


class RerunClass(str, Enum):
    STILL_TRUE = "still_true"
    DRIFTED = "drifted"
    WRONG_AT_PUBLISH = "wrong_at_publish"
    NOT_RERUNNABLE = "not_rerunnable"


# Words that, as a shell token, mean "this command may talk to the network or
# change state". The denylist is a refusal list, not a sandbox: a command that
# passes it is still run with a timeout under `nice`, and the caller chooses cwd.
DENYLIST: frozenset[str] = frozenset(
    {
        # network
        "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "rsync", "telnet", "ftp",
        "pip", "pip3", "npm", "npx", "yarn", "pnpm", "apt", "apt-get", "brew", "docker",
        "fetch", "ping", "dig", "nslookup",
        # write verbs
        "rm", "mv", "cp", "dd", "tee", "truncate", "touch", "mkdir", "rmdir", "ln", "chmod",
        "chown", "install", "sed", "perl", "kill", "pkill", "killall", "shred", "unlink",
        "sudo", "su", "doas", "crontab", "systemctl", "service", "reboot", "shutdown",
    }
)
# git and similar tools are allowed only with read-only subcommands.
_SUBCOMMAND_ALLOW: dict[str, frozenset[str]] = {
    "git": frozenset({"log", "show", "grep", "status", "diff", "rev-parse", "ls-files", "cat-file", "branch", "tag", "describe"}),
}
_WRITE_REDIRECT = re.compile(r"(^|[^<>])>{1,2}(?!&[12])")  # `>` or `>>` not part of `2>&1`
_NETWORK_HINT = re.compile(r"https?://|wss?://|ftp://", re.IGNORECASE)
_SED_INPLACE = re.compile(r"\bsed\b[^|;&]*\s-i")
_PY_WRITE = re.compile(r"open\([^)]*['\"][wa][b+]*['\"]")


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=True)
    except ValueError:
        return cmd.split()


def refusal_reason(cmd: str) -> Optional[str]:
    """Return a named reason to refuse ``cmd``, or ``None`` if it may run.

    This is conservative on purpose. Inline scripts (``python -c``, ``node -e``)
    are inspected for obvious write/network patterns but a dedicated reviewer is
    still the only real guarantee.
    """
    if not cmd or not cmd.strip():
        return "no_command_recorded"
    if _NETWORK_HINT.search(cmd):
        return "denylist:network_url"
    if _WRITE_REDIRECT.search(cmd):
        return "denylist:output_redirect"
    if _SED_INPLACE.search(cmd):
        return "denylist:sed_inplace"
    if _PY_WRITE.search(cmd):
        return "denylist:inline_file_write"
    toks = _tokens(cmd)
    # look at every token so `a | rm b` and `cd x && curl y` are caught.
    for i, tok in enumerate(toks):
        base = tok.rsplit("/", 1)[-1]
        if base in DENYLIST:
            return f"denylist:{base}"
        if base in _SUBCOMMAND_ALLOW:
            sub = next((t for t in toks[i + 1 :] if not t.startswith("-")), "")
            if sub not in _SUBCOMMAND_ALLOW[base]:
                return f"denylist:{base}_{sub or 'no_subcommand'}"
        if base in {"http.request", "requests.get", "urllib.request"}:
            return "denylist:network_call"
    if re.search(r"\b(requests|urllib|http\.client|socket|fetch)\b", cmd):
        return "denylist:network_library"
    return None


@dataclass(frozen=True)
class Adjudication:
    """A human-written verdict on a specific mismatch, with evidence."""

    cls: RerunClass
    evidence: str
    strict: bool = False
    lenient: bool = False


@dataclass(frozen=True)
class RerunResult:
    seq: Any
    cls: RerunClass
    reason: str
    observed: str
    rerun: Optional[str] = None
    strict: bool = False
    lenient: bool = False
    ms: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def checked(self) -> bool:
        return self.cls is not RerunClass.NOT_RERUNNABLE

    @property
    def drift(self) -> bool:
        return self.cls is RerunClass.DRIFTED


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def _run_shell(cmd: str, timeout: float, cwd: Optional[str], nice: int) -> tuple[Optional[str], Optional[str], int]:
    import time

    argv = ["nice", "-n", str(nice), "sh", "-c", cmd]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
    except subprocess.TimeoutExpired:
        return None, f"timeout_after_{timeout:g}s", int((time.monotonic() - t0) * 1000)
    except OSError as exc:
        return None, f"spawn_failed:{exc.__class__.__name__}", int((time.monotonic() - t0) * 1000)
    ms = int((time.monotonic() - t0) * 1000)
    if proc.returncode != 0:
        return proc.stdout, f"exit_{proc.returncode}", ms
    return proc.stdout, None, ms


def rerun(
    record: Mapping[str, Any],
    *,
    timeout: float = 30.0,
    cwd: Optional[str] = None,
    nice: int = 10,
    adjudication: Optional[Adjudication] = None,
    callables: Optional[Mapping[str, Callable[[], Any]]] = None,
    compare: Optional[Callable[[str, str], bool]] = None,
) -> RerunResult:
    """Re-run one claim record and classify it.

    ``record`` needs ``check`` (shell command or ``callable:<name>`` resolved via
    ``callables``) and ``observed``. Optional: ``seq``, ``expected`` (what the
    claim asserted; enables the record-consistency check), ``snapshot`` (truthy
    means a mismatch is drift by construction - use sparingly and say why in
    ``snapshot_reason``).
    """
    seq = record.get("seq")
    observed = _norm(record.get("observed"))
    check = str(record.get("check") or record.get("cmd") or "")
    expected = record.get("expected")
    eq = compare or (lambda a, b: _norm(a) == _norm(b))

    # Record-consistency first: this needs no execution and catches claims that
    # disagreed with their own output at publish time.
    if expected is not None and not eq(str(expected), observed):
        adj = adjudication
        return RerunResult(
            seq, RerunClass.WRONG_AT_PUBLISH,
            "record_inconsistent: claim asserted a value its own check did not produce at the time",
            observed, strict=True, lenient=bool(adj.lenient) if adj else False,
        )

    # Execute.
    rerun_value: Optional[str]
    ms: Optional[int] = None
    if check.startswith("callable:"):
        name = check[len("callable:"):]
        fn = (callables or {}).get(name)
        if fn is None:
            return RerunResult(seq, RerunClass.NOT_RERUNNABLE, f"callable_not_provided:{name}", observed)
        try:
            rerun_value = _norm(fn())
        except Exception as exc:  # noqa: BLE001 - the reason is the point
            return RerunResult(seq, RerunClass.NOT_RERUNNABLE, f"callable_raised:{exc.__class__.__name__}", observed)
    else:
        why = refusal_reason(check)
        if why:
            return RerunResult(seq, RerunClass.NOT_RERUNNABLE, why, observed)
        out, err, ms = _run_shell(check, timeout, cwd, nice)
        if err is not None and out is None:
            return RerunResult(seq, RerunClass.NOT_RERUNNABLE, err, observed, ms=ms)
        if err is not None:
            # non-zero exit: only trust it if the output still matches exactly.
            if eq(_norm(out), observed):
                return RerunResult(seq, RerunClass.STILL_TRUE, f"output_matches_despite_{err}", observed, _norm(out), ms=ms)
            return RerunResult(seq, RerunClass.NOT_RERUNNABLE, f"check_failed:{err}", observed, _norm(out), ms=ms)
        rerun_value = _norm(out)

    if eq(rerun_value, observed):
        # an adjudicated strict flag survives a matching re-run (stale restatement).
        if adjudication and adjudication.cls is RerunClass.WRONG_AT_PUBLISH:
            return RerunResult(seq, RerunClass.WRONG_AT_PUBLISH, f"adjudicated: {adjudication.evidence}", observed, rerun_value,
                               strict=adjudication.strict, lenient=adjudication.lenient, ms=ms)
        return RerunResult(seq, RerunClass.STILL_TRUE, "rerun_matches_observed", observed, rerun_value, ms=ms)

    # Mismatch. Needs a named reason from a human or a declared snapshot.
    if adjudication is not None:
        return RerunResult(seq, adjudication.cls, f"adjudicated: {adjudication.evidence}", observed, rerun_value,
                           strict=adjudication.strict, lenient=adjudication.lenient, ms=ms)
    if record.get("snapshot"):
        return RerunResult(seq, RerunClass.DRIFTED, f"declared_snapshot: {record.get('snapshot_reason') or 'no reason given'}",
                           observed, rerun_value, ms=ms)
    return RerunResult(seq, RerunClass.NOT_RERUNNABLE, f"mismatch_unadjudicated: rerun={rerun_value!r} observed={observed!r}",
                       observed, rerun_value, ms=ms)


def rerun_many(
    records: Sequence[Mapping[str, Any]],
    *,
    adjudications: Optional[Mapping[Any, Adjudication]] = None,
    **kwargs: Any,
) -> list[RerunResult]:
    adj = adjudications or {}
    return [rerun(r, adjudication=adj.get(r.get("seq")), **kwargs) for r in records]
