"""Unavailable or invalid evidence cannot become a passing policy."""
import json

import numpy as np
import pytest

from pymc_monitor import convergence as cv


def draws(chains=4):
    rng = np.random.default_rng(1)
    return cv.draws_from_arrays({"x": rng.normal(size=(chains, 600))},
        diverging=np.zeros((chains, 600)), energy=rng.normal(size=(chains, 600)))


def test_single_chain_rhat_must_be_explicitly_disabled():
    d = draws(1)
    target = cv.normalize_target({"preset": "draft", "ess": 1})
    assessment = cv.assess(d, target)
    assert not assessment["met"] and assessment["checks"]["r_hat"]["ok"] is False
    assert "unavailable" in cv.summarize(assessment)
    assert cv.diagnose(d, assessment)[0] == "diagnostics_unavailable"
    assert cv.assess(d, {**target, "r_hat": None})["met"]


def test_missing_bfmi_on_one_chain_fails():
    d = draws()
    d.energy[0, :] = np.nan
    a = cv.assess(d, cv.normalize_target(True))
    assert not a["met"] and a["checks"]["bfmi"]["ok"] is False
    cause, hint = cv.diagnose(d, a)
    assert cause == "diagnostics_unavailable" and "bfmi" in hint


def test_constant_free_variable_fails():
    d = cv.draws_from_arrays({"x": np.ones((4, 500))})
    a = cv.assess(d, cv.normalize_target("draft"))
    assert not a["met"] and not a["checks"]["finite_variation"]["ok"]
    report = cv.make_report(a, status="complete", target=None)
    json.dumps(report, allow_nan=False)
    assert report["checks"]["r_hat"]["value"] is None


def test_missing_stats_and_treedepth_gate():
    d = draws()
    d.maxdepth_frac = .5
    a = cv.assess(d, cv.normalize_target({"treedepth": .1}))
    assert not a["met"] and cv.diagnose(d, a)[0] == "treedepth"
    div = np.zeros((4, 600))
    div[0, 0] = np.nan
    d2 = cv.draws_from_arrays({"x": d.values}, diverging=div)
    assert d2.diverging is None
    assert not cv.assess(d2, cv.normalize_target(True))["met"]


@pytest.mark.parametrize("target", [42, {"preset": "missing"}, {"quantiles": 1},
    {"divergences": -1}, {"escalate": "yes"}, {"superchains": 2.5}])
def test_bad_policy_rejected(target):
    with pytest.raises((ValueError, TypeError)):
        cv.normalize_target(target)


def test_bad_arrays_and_membership():
    with pytest.raises(ValueError, match="no variables"):
        cv.draws_from_arrays({})
    with pytest.raises(ValueError, match="expected"):
        cv.draws_from_arrays({"x": [1, 2]})
    with pytest.raises(ValueError, match="one entry"):
        cv.nested_rhat(np.ones((4, 20, 1)), [0, 1])
    d = draws()
    d.superchain_ids = [0, 0, 0, 1]
    with pytest.raises(ValueError, match="equal size"):
        cv.assess(d, cv.normalize_target(True))


def test_display_only_reports_non_gating_rhat():
    a = cv.assess(draws(), cv.normalize_target({"r_hat": None}))
    assert a["checks"]["r_hat"]["ok"] is None
    assert "not gating" in cv.summarize(a)
    assert cv.read_report({}) is None
    assert cv.verdict_message({"status": "cap", "met": True}).startswith("draw cap reached, targets met")
    assert "stopped by user" in cv.verdict_message({"status": "stopped"})
    assert "unreachable" in cv.verdict_message({"status": "futile"})
