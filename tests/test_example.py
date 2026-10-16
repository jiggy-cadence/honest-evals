"""The worked example must reproduce its headline and pass its own canary."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "fixtures" / "claim_outcomes.jsonl"


def test_fixture_is_text_free_and_counts_match() -> None:
    rows = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    assert len(rows) == 99
    assert all("claim" not in r and "cmd" not in r and "check" not in r for r in rows)
    a = [r for r in rows if r["population"] == "logged_at_claim"]
    b = [r for r in rows if r["population"] == "chosen_for_publication"]
    assert (len(a), len(b)) == (17, 82)


def test_example_reproduces_and_canary_passes() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "selection_effect.py")], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "MATCH logged_at_claim strict: expected 6/16 CI 18.5-61.4" in out
    assert "MATCH chosen_for_publication strict: expected 3/77 CI 1.3-10.8" in out
    assert "canary PASS" in out and "STABLE" in out and "FRAGILE" not in out
