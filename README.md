# honest-evals

Tools for evaluations that have to prove they can fail.

An evaluation that cannot fail is not measuring anything. This package is five
small, dependency-light pieces for the parts of an eval pipeline where that
usually goes wrong: the denominator gets chosen after the verdicts are known,
the checker silently stops checking, the threshold is tuned until the story
holds, and the record of what was claimed can be quietly edited.

## Why

We built this to answer one reviewer question about an AI agent's self-reports:
*errors per auditable claim, with a known denominator*. Answering it needed
four things the usual eval stack does not give you:

1. A record of claims written **before** their verdicts, so the denominator
   cannot be curated (`ledger`), with truncation detectable, not just edits.
2. A way to re-execute each claim's check safely and to say *why* a row could
   not be re-run, instead of dropping it (`rerun`).
3. Statistics that treat two differently-selected populations as two
   populations, with intervals that are honest at n=16 (`selection`).
4. Proof that the pipeline would have caught an error if there had been one
   (`canary`), and that the headline does not depend on a knob (`cutoffs`).

The worked example in `examples/` reproduces a real result from an agent's own
records: 6/16 = 37.5% of claims logged at utterance were wrong, versus 3/77 =
3.9% of claims the agent chose to publish with a check. The 10x gap is the
selection effect, and `selection_effect.py` refuses to print it until the
canary passes and the cutoff sweep is stable.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest
python examples/selection_effect.py
```

Dependencies: numpy, scipy, pandas. Tests: pytest. Python >= 3.10.

## Quickstart

### ledger: append-only, hash-chained claims

```python
from honest_evals import Ledger, verify_chain

led = Ledger("claims.jsonl")
led.append("tests pass: 67", "pytest -q | tail -1", "67 passed", kind="numeric")
rep = verify_chain(led)        # checks prev links, every sha256, and the head file
assert rep.ok, rep.reason      # "hash mismatch (record edited...)", "truncated...", etc.
```

Each record carries `prev` and `hash` (sha256 over every other field, including
`meta`). A sidecar `claims.jsonl.head` records the last hash; a shortened file
still has valid internal links, so the head check is what catches truncation.
`append` refuses to extend a broken chain.

### rerun: re-execute read-only checks, name every refusal

```python
from honest_evals import rerun, Adjudication, RerunClass

r = rerun({"seq": 1, "check": "wc -l < data.jsonl", "observed": "830"}, timeout=10)
r.cls      # still_true | drifted | wrong_at_publish | not_rerunnable
r.reason   # "rerun_matches_observed", "denylist:curl", "timeout_after_10s",
           # "mismatch_unadjudicated: rerun='842' observed='830'", ...
```

Commands run under `nice` with a timeout. Anything with a URL, an output
redirect, a network tool, a write verb, `git push`/`fetch`, or an inline
`open(..., 'w')` is refused with a named reason. A mismatch is **never**
auto-classified as drift: you either pass an `Adjudication` with evidence, or
mark the record `snapshot=True` with a `snapshot_reason`, or it stays
unresolved. If the record carries `expected` (what the claim asserted) and it
disagrees with `observed` (what the check printed at the time), the row is
`wrong_at_publish` without running anything.

### selection: two populations, two cutoffs, two intervals

```python
from honest_evals import outcomes_from_records, selection_effect

eff = selection_effect(outcomes_from_records(logged_rows), outcomes_from_records(published_rows))
print(eff.report())            # strict/lenient x drift excluded/included
eff.cells[("strict", False)]   # (RateCell, RateCell): rate, wilson, bootstrap, best/worst bound
eff.tests[("strict", False)]   # two-proportion z + Fisher exact
eff.to_frame()                 # pandas DataFrame, 8 rows
```

### canary: no "clean" without a caught plant

```python
from honest_evals import canary, Canary, CanaryFailed

@canary(false_item={"id": -1, "bad": True}, true_item={"id": -2, "bad": False}, key=lambda r: r["id"])
def find_bad(rows):
    return [r["id"] for r in rows if r["bad"]]

find_bad(rows)   # runs the canary on a copy first; raises CanaryFailed if the
                 # false plant is missed or the true plant is flagged
```

Or as a context manager: `with Canary(rows, false_item=..., true_item=..., key=...) as c:
c.observe(find_bad(c.data))`. Leaving the block without a passing observation raises.

### cutoffs: sweep the knob or do not report

```python
from honest_evals import sweep, FragileFinding

finding = sweep([0.01, 0.05, 0.10], lambda alpha: rank_strategies(alpha))
if isinstance(finding, FragileFinding):
    print(finding.summary())   # "FRAGILE: winner depends on the threshold - 'B' wins at [0.01]; 'A' wins at [0.05, 0.1]"
else:
    finding.winner
```

Requires at least three distinct settings. `evaluate` returns a verdict or a
ranking; the winner must be the same everywhere or you get a `FragileFinding`
with no `winner`.

## Honest limits

- **The denylist is a refusal list, not a sandbox.** It catches the obvious
  (URLs, redirects, `rm`, `curl`, `git push`, inline file writes) and will miss
  a determined command. Run re-runs in a throwaway checkout if the commands are
  not yours.
- **A re-run of your own check reproduces your own wrongness.** If the recorded
  command measures something adjacent to the claim, `still_true` means the
  command still prints the same thing, not that the claim is true. The example
  data had four such rows; they are only found by reading.
- **Drift adjudication is a human judgement** and the package makes you write
  it down, but it cannot check it. Unadjudicated mismatches stay unresolved on
  purpose.
- **The ledger proves the record was not altered after writing**, not that the
  record was complete. Claims never logged are invisible to it. Truncation is
  only detectable if the attacker does not also rewrite the head file; keep the
  head file (or its hash) somewhere else if that matters.
- **Canaries prove the checker can catch the plant you chose.** Choose plants
  that look like real failures, and more than one over time.
- **Small n.** Wilson and bootstrap intervals are reported because at n=16 the
  point estimate is nearly meaningless; quote the interval.
- The selection-effect test compares detected-by-self error rates between two
  records. It is not an estimate of ground-truth error.

## Layout

```
src/honest_evals/   ledger.py rerun.py selection.py canary.py cutoffs.py
tests/              one file per module, plus the worked example
examples/           selection_effect.py, README.md walkthrough, fixtures/claim_outcomes.jsonl
```

## License

MIT.
