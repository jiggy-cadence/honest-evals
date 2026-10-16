import pytest

from honest_evals.canary import Canary, CanaryFailed, canary, run_canary

ROWS = [{"seq": 1, "strict": False}, {"seq": 2, "strict": False}, {"seq": 3, "strict": True}]
BAD = {"seq": -1, "strict": True}
GOOD = {"seq": -2, "strict": False}


def key(r: dict) -> int:
    return r["seq"]


def real_checker(rows):
    return [r["seq"] for r in rows if r["strict"]]


def noop_checker(rows):
    return []


def flag_everything(rows):
    return [r["seq"] for r in rows]


def test_real_checker_passes() -> None:
    v = run_canary(real_checker, ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert v.passed and v.caught_false and v.passed_true and v.flagged_on_copy == 2


def test_noop_checker_fails() -> None:
    """Negative control: a checker that flags nothing must not be allowed to say clean."""
    v = run_canary(noop_checker, ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert not v.passed and not v.caught_false and v.passed_true and "not caught" in v.reason


def test_flag_everything_fails() -> None:
    v = run_canary(flag_everything, ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert not v.passed and v.caught_false and not v.passed_true and "indiscriminately" in v.reason


def test_inverted_checker_fails() -> None:
    v = run_canary(lambda rows: [r["seq"] for r in rows if not r["strict"]], ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert not v.passed and "inverted" in v.reason


def test_crashing_checker_fails() -> None:
    def boom(rows):
        raise RuntimeError("nope")

    v = run_canary(boom, ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert not v.passed and "raised RuntimeError" in v.reason


def test_original_data_untouched() -> None:
    rows = [dict(r) for r in ROWS]
    run_canary(real_checker, rows, false_item=BAD, true_item=GOOD, key=key)
    assert rows == ROWS


def test_checker_may_return_items_instead_of_keys() -> None:
    v = run_canary(lambda rows: [r for r in rows if r["strict"]], ROWS, false_item=BAD, true_item=GOOD, key=key)
    assert v.passed


def test_decorator_passes_through_real_result() -> None:
    guarded = canary(false_item=BAD, true_item=GOOD, key=key)(real_checker)
    assert guarded(ROWS) == [3]
    assert guarded.canary_verdict(ROWS).passed


def test_decorator_refuses_noop() -> None:
    guarded = canary(false_item=BAD, true_item=GOOD, key=key)(noop_checker)
    with pytest.raises(CanaryFailed) as ei:
        guarded(ROWS)
    assert not ei.value.verdict.caught_false


def test_context_manager_pass_and_fail() -> None:
    with Canary(ROWS, false_item=BAD, true_item=GOOD, key=key) as c:
        assert len(c.data) == len(ROWS) + 2
        c.observe(real_checker(c.data))
    assert c.verdict is not None and c.verdict.passed

    with pytest.raises(CanaryFailed):
        with Canary(ROWS, false_item=BAD, true_item=GOOD, key=key) as c:
            c.observe(noop_checker(c.data))

    with pytest.raises(CanaryFailed, match="never reported"):
        with Canary(ROWS, false_item=BAD, true_item=GOOD, key=key):
            pass


def test_context_manager_does_not_mask_body_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        with Canary(ROWS, false_item=BAD, true_item=GOOD, key=key):
            1 / 0
