"""Public API, real sampler, control, policy, and streaming regressions."""
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import numpy as np
import pymc as pm
import pytest

import pymc_monitor as monitor
from pymc_monitor import convergence as cv
from pymc_monitor.dashboard import Dashboard, DashboardState
from pymc_monitor.progress import ConvergenceUpdate, JobPhase, PhaseUpdate


@pytest.fixture
def model():
    with pm.Model() as model:
        pm.HalfNormal("scale")
        pm.Normal("obs", 0, observed=[0.1, -0.3])
    return model


def test_api_native_constrained_report_and_events(model):
    events = []
    result = monitor.sample(model=model, draws=12, tune=10, chains=2, cores=1,
                            progress=False, nuts_sampler="pymc", until={"preset": "draft", "ess": 5},
                            on_event=events.append)
    report = monitor.read_report(result)
    assert report["status"] in ("cap", "met", "futile")
    assert report["draws"] == result.posterior.sizes["draw"]
    assert any(isinstance(event, ConvergenceUpdate) and event.final for event in events)
    assert np.all(result.posterior["scale"].values >= 0)
    assert cv.check_convergence(result, "draft")["chains"] == 2


def test_iter_sample_returns_result(model):
    stream = monitor.iter_sample(model, {"draws": 8, "tune": 5, "chains": 2, "cores": 1}, nuts_sampler="pymc")
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as end:
            result = end.value
            break
    assert isinstance(result, monitor.SamplingResult)
    assert monitor.read_report(result.idata) == result.report
    assert events[0].phase == JobPhase.SAMPLING
    assert events[-2].status == "done"
    assert events[-1].status == result.report["status"]
    assert result.effective_sampler == "pymc"


def test_start_wait_snapshot(model):
    handle = monitor.start(model=model, draws=7, tune=5, chains=2, cores=1, progress=False)
    try:
        result = handle.wait(timeout=90)
        assert handle.done
        assert handle.snapshot()["report"] == monitor.read_report(result)
        assert handle.dashboard_url is None
    finally:
        handle.close()


def test_validation_prevents_silent_bad_policies(model):
    for key, val in {"ess": 0, "mcse": -1, "growth": float("nan"),
                     "patience": 0, "escalate": -1, "bfmi": float("inf"),
                     "r_hat": 0.9, "superchains": 1, "var_names": []}.items():
        with pytest.raises((ValueError, TypeError), match="until"):
            monitor.normalize_target({key: val})
    with pytest.raises(ValueError, match="JAX"):
        next(monitor.iter_sample(model, {}, nuts_sampler="numpyro", until=True))
    with pytest.raises(ValueError, match="superchains"):
        next(monitor.iter_sample(model, {"chains": 3}, until={"superchains": 2}))
    with pytest.raises(ValueError, match="callback"):
        next(monitor.iter_sample(model, {"callback": "not callable"}))


def test_missing_required_hmc_stats_fail_closed():
    d = cv.draws_from_arrays({"x": np.random.default_rng(4).normal(size=(4, 200))})
    a = cv.assess(d, cv.normalize_target(True))
    assert not a["met"]
    assert a["checks"]["divergences"]["ok"] is False
    assert a["checks"]["bfmi"]["ok"] is False


def test_normalizer_protocol_no_cache_dependency(model):
    norm = monitor.sample.__pymc_cache_normalize__(model, {"draws": 10, "progress": False,
                                                        "until": "draft", "nuts_sampler": "pymc"})
    assert set(norm) == {"call_kwargs", "key_kwargs", "sampler_identity", "versions"}
    assert "model" not in norm["call_kwargs"]
    assert norm["key_kwargs"]["until"]["ess"] == 100
    assert "progress" not in norm["key_kwargs"]
    assert norm["sampler_identity"] == "pymc"
    assert norm["versions"]["pymc-monitor"] == "0.1.0"
    with pytest.raises(ValueError, match="cacheable"):
        monitor.sample.__pymc_cache_normalize__(model, {"control": monitor.RunControl()})


def test_dashboard_auth_and_local_stop():
    control = monitor.RunControl()
    state = DashboardState()
    service = Dashboard()
    run_id = service.add_run(state, control)
    try:
        url = service.url(run_id)
        assert urlsplit(url).hostname == "127.0.0.1"
        token = urlsplit(url).path.split("/")[2]
        base = f"http://{urlsplit(url).netloc}/runs/{run_id}"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/state")
        assert exc.value.code == 403
        req = urllib.request.Request(base + "/state", headers={"Authorization": f"Bearer {token}"})
        assert b'"complete":false' in urllib.request.urlopen(req).read()
        state.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "working", 1))
        assert state.snapshot()["phases"][0]["detail"] == "working"
        bad = urllib.request.Request(base + "/stop", method="POST",
            headers={"Authorization": f"Bearer {token}", "Origin": "https://evil.test"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad)
        assert exc.value.code == 403
        assert not control.stopped
        good = urllib.request.Request(base + "/stop", method="POST",
            headers={"Authorization": f"Bearer {token}"})
        assert urllib.request.urlopen(good).status == 204
        assert control.stopped
    finally:
        service.close()


def test_reducer_schema_matches_cloud_consumer():
    state = DashboardState()
    state.show_convergence(ConvergenceUpdate({}, draws=20,
                          verdict={"decision": "met"}, final=True))
    state.finish()
    assert state.snapshot()["convergence"]["verdict"]["final"]
    assert state.snapshot()["complete"]


def test_nutpie_stream(model):
    pytest.importorskip("nutpie")
    result = monitor.sample(model=model, nuts_sampler="nutpie", draws=8, tune=8,
                            chains=2, progress=False)
    assert monitor.read_report(result)["status"] == "complete"
