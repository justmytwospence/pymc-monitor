"""Lifecycle and dispatch tests, including real constrained and partial traces."""
import threading
import time

import numpy as np
import pymc as pm
import pytest

import pymc_monitor as monitor
from pymc_monitor import engine, request
from pymc_monitor.control import RunControl, SamplingStoppedError
from pymc_monitor.events import ConvergenceUpdate, PhaseUpdate, SamplingProgress
from pymc_monitor.idata import get_group


def drain(stream):
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as end:
            return events, end.value


@pytest.fixture
def model():
    with pm.Model() as model:
        scale = pm.HalfNormal("scale", 2)
        x = pm.Normal("x", sigma=scale)
        pm.Deterministic("twice", 2 * x)
    return model


@pytest.fixture
def idata():
    import arviz as az

    rng = np.random.default_rng(3)
    groups = {"posterior": {"scale": np.abs(rng.normal(size=(2, 100))),
                            "x": rng.normal(size=(2, 100))},
              "sample_stats": {"energy": rng.normal(size=(2, 100)),
                               "diverging": np.zeros((2, 100), bool)}}
    try:
        return az.from_dict(**groups)
    except TypeError:
        return az.from_dict(groups)


def test_control_external_poll_latches_and_warns(monkeypatch):
    calls = []
    control = RunControl(poll=lambda: calls.append(1) or True)
    assert control.poll() and control.stopped
    assert control.poll() and calls == [1]
    with pytest.raises(TypeError, match="stop_requested"):
        RunControl(stop_requested=3)
    clock = [0.0]
    monkeypatch.setattr("pymc_monitor.control.time.monotonic", lambda: clock[0])
    calls = []
    def fail():
        calls.append(True)
        raise RuntimeError("lost control channel")
    control = RunControl(stop_requested=fail)
    with pytest.warns(RuntimeWarning, match="lost control channel"):
        assert not control.is_stop_requested()
    assert not control.poll() and len(calls) == 1
    clock[0] = 0.51
    assert not control.poll() and len(calls) == 2
    control.request_stop()
    assert control.stopped and control.is_stop_requested()


def test_defaults_and_native_constraints(model, monkeypatch):
    monkeypatch.setattr(request, "find_spec", lambda _: object())
    monkeypatch.setattr(request, "pymc_major", lambda: 6)
    assert request.resolve(model, {}).sampler == "nutpie"
    assert request.resolve(model, {}).settings["tune"] == 400
    monkeypatch.setattr(request, "pymc_major", lambda: 5)
    assert request.resolve(model, {}).settings["tune"] == 1000
    assert request.resolve(model, {"callback": lambda **_: None}).sampler == "pymc"
    assert request.resolve(model, {"step": object()}).sampler == "pymc"
    monkeypatch.setattr(request, "find_spec", lambda _: None)
    assert request.resolve(model, {}).sampler == "pymc"
    with pm.Model() as discrete:
        pm.Bernoulli("z", 0.5)
    monkeypatch.setattr(request, "find_spec", lambda _: object())
    assert request.resolve(discrete, {}).sampler == "pymc"


@pytest.mark.parametrize("kwargs,options", [
    ({"draws": 0}, {}), ({"draws": True}, {}), ({"tune": -1}, {}),
    ({"cores": 0}, {}), ({"chains": False}, {}), ({"progressbar": True}, {}),
    ({"tune": 0}, {"nuts_sampler": "nutpie"}),
    ({"return_inferencedata": False}, {}), ({"until": True}, {}),
    ({"nuts_sampler": "pymc"}, {}), ({}, {"nuts_sampler": "bad"}),
    ({}, {"initial_strategy": "bad"}), ({}, {"control": object()}),
    ({"callback": 1}, {}), ({"callback": lambda **_: None}, {"nuts_sampler": "nutpie"}),
    ({"step": object()}, {"nuts_sampler": "nutpie"}),
    ({"trace": object()}, {}), ({"var_names": ["x"]}, {}),
    ({"initvals": {}}, {"initial_strategy": "prior"}),
    ({"chains": 2}, {"until": {"superchains": 2}}),
    ({"initvals": {}}, {"until": {"superchains": 2}}),
    ({}, {"until": True, "nuts_sampler": "blackjax"}),
    ({}, {"until": {"var_names": "absent"}}),
])
def test_request_validation(model, kwargs, options):
    with pytest.raises((ValueError, TypeError)):
        request.resolve(model, kwargs, **options)


def test_model_context_resolved_before_thread():
    with pm.Model():
        pm.Normal("x")
        handle = monitor.start(draws=5, tune=5, chains=2, cores=1,
                               nuts_sampler="pymc", progress=False)
    with handle:
        assert handle.wait(60).posterior.sizes["draw"] == 5
        assert handle.snapshot()["state"]["complete"]
    handle.close()


def test_constrained_converter_is_not_prior_sampling(model):
    converter = engine._live_converter(model, ["scale", "x"])
    point = {"scale_log__": np.log(7.0), "x": np.array(-4.0)}
    for _ in range(3):
        actual = converter(point)
        assert actual["scale"] == pytest.approx(7.0)
        assert actual["x"] == -4
    with pytest.raises(ValueError, match="unknown"):
        engine._live_converter(model, ["absent"])
    with pm.Model() as hierarchical:
        lower = pm.Normal("lower")
        pm.Uniform("u", lower=lower, upper=lower + 2)
    actual = engine._live_converter(hierarchical, ["u"])(
        {"lower": np.array(5.0), "u_interval__": np.array(0.0)})
    assert actual["u"] == pytest.approx(6.0)


def test_native_user_callback_and_exact_live_values(model):
    points = []
    def callback(*, trace, draw):
        if not draw.tuning:
            points.append((draw.chain, np.exp(draw.point["scale_log__"])))
    events, result = drain(engine.iter_sample(model, {"draws": 10, "tune": 10,
        "chains": 2, "cores": 1, "random_seed": 7, "callback": callback}))
    assert len(points) == 20
    for chain in range(2):
        np.testing.assert_allclose(result.idata.posterior.scale.values[chain],
                                   [x for c, x in points if c == chain])
    assert any(isinstance(event, SamplingProgress) for event in events)
    assert isinstance(events[-3], ConvergenceUpdate) and events[-3].final


@pytest.mark.parametrize("sampler", ["numpyro", "blackjax"])
def test_jax_phase_only_forwarding_and_stop(model, idata, sampler):
    control = RunControl()
    captured = {}
    def sample_fn(**kwargs):
        captured.update(kwargs)
        control.stop()
        time.sleep(0.15)
        return idata
    events, result = drain(engine.iter_sample(model, {}, nuts_sampler=sampler,
                                              control=control, sample_fn=sample_fn))
    assert captured["nuts_sampler"] == sampler
    assert "callback" not in captured and captured["progressbar"] is False
    assert not any(isinstance(event, SamplingProgress) for event in events)
    assert any("cannot abort" in e.message for e in events if isinstance(e, PhaseUpdate))
    assert result.report["status"] == "complete"  # requested stop did not abort JAX


def test_prestart_stop_and_sampling_errors(model):
    control = RunControl()
    control.stop()
    with pytest.raises(SamplingStoppedError, match="before it started"):
        drain(engine.iter_sample(model, {}, control=control))
    def failed(**_):
        raise RuntimeError("sampler failure")
    with pytest.raises(RuntimeError, match="sampler failure"):
        drain(engine.iter_sample(model, {}, nuts_sampler="pymc", sample_fn=failed))
    with pytest.raises(TypeError, match="sample_fn"):
        next(engine.iter_sample(model, {}, sample_fn=1))


def test_generator_close_joins_worker(model, idata):
    control = RunControl()
    done = threading.Event()
    def fake(**_):
        while not control.stopped:
            time.sleep(0.01)
        done.set()
        return idata
    stream = engine.iter_sample(model, {}, nuts_sampler="numpyro", sample_fn=fake, control=control)
    next(stream)
    next(stream)  # phase-only announcement, before worker starts
    # Closing here must not leave a sampler thread behind.
    stream.close()
    assert control.stopped
    assert not done.is_set()


def test_event_callback_error_stops_run(model, monkeypatch):
    control = RunControl()
    def fail(event):
        raise RuntimeError("consumer failed")
    handle = monitor.start(model=model, draws=10, progress=False, control=control, on_event=fail)
    with pytest.raises(RuntimeError, match="consumer failed"):
        handle.wait(30)
    handle.close()
    assert handle.done and control.stopped
    with pytest.raises(TypeError, match="on_event"):
        monitor.start(model=model, on_event=42)


def test_cache_normalizes_execution_and_rejects_callback(model):
    normalizer = monitor.sample.__pymc_cache_normalize__
    one = normalizer(model, {"nuts_sampler": "pymc", "until": True, "cores": 1})
    two = normalizer(model, {"nuts_sampler": "pymc", "until": "final", "cores": 1,
                             "draws": 1000, "tune": 1000, "chains": 2,
                             "dashboard": True, "progress": False})
    assert set(one) == {"call_kwargs", "key_kwargs", "sampler_identity", "versions"}
    assert one["key_kwargs"] == two["key_kwargs"]
    assert "model" not in one["call_kwargs"]
    with pytest.raises(ValueError, match="cacheable"):
        normalizer(model, {"callback": lambda **_: None})


def test_native_adaptive_balanced_recovery():
    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Deterministic("twice", 2*x)
    events, result = drain(engine.iter_sample(model, {"draws": 3000, "tune": 100,
        "chains": 2, "cores": 1, "random_seed": 12}, nuts_sampler="pymc",
        until={"preset": "draft", "ess": 10, "r_hat": 1.5, "patience": 1, "futility": False}))
    post = get_group(result.idata, "posterior")
    assert post.sizes["chain"] == 2
    assert 4 <= post.sizes["draw"] < 3000
    np.testing.assert_allclose(post.twice, 2 * post.x)
    assert result.report["status"] == "met"
    assert any(isinstance(e, ConvergenceUpdate) and not e.final for e in events)


def test_native_early_stop_without_balanced_draws(model):
    control = RunControl()
    def stop(**_):
        control.stop()
    with pytest.raises(SamplingStoppedError, match="complete posterior"):
        monitor.sample(model=model, draws=10, tune=5, chains=2, cores=1,
                       callback=stop, control=control, progress=False)


def test_live_failure_is_not_silenced(model, monkeypatch):
    def fail(*_):
        raise ValueError("broken diagnostic")
    monkeypatch.setattr(engine.cv.Controller, "step", fail)
    with pytest.raises(RuntimeError, match="live diagnostics failed"):
        monitor.sample(model=model, draws=3000, tune=50, chains=2, cores=1,
                       nuts_sampler="pymc", until="draft", progress=False)


def test_final_recovered_report_overrules_checkpoint(model, idata):
    controller = engine.cv.Controller(engine.cv.normalize_target(True), 2, 100, stop=True)
    report, event = engine._assessment(idata, ["x", "scale"], None, controller.target,
                                        controller, "met", "met", [])
    assert report["status"] == "cap" and not report["met"]
    assert event.verdict["decision"] == "cap"
    with pytest.raises(SamplingStoppedError, match="membership"):
        engine._assessment(idata, ["x"], [0, 0, 1, 1], controller.target,
                           controller, "met", "met", [])


def test_nutpie_real_adaptive_stream():
    pytest.importorskip("nutpie")
    with pm.Model() as model:
        pm.Normal("x")
    events, result = drain(engine.iter_sample(model, {"draws": 100000, "tune": 100,
        "chains": 2, "random_seed": 4, "target_accept": .9}, nuts_sampler="nutpie",
        until={"preset": "draft", "ess": 10, "r_hat": 1.5, "patience": 1, "futility": False}))
    assert 4 <= result.report["draws"] < 100000
    assert result.report["status"] == "met"
    assert any(isinstance(e, ConvergenceUpdate) and not e.final for e in events)


@pytest.mark.parametrize("sampler", ["pymc", "nutpie"])
def test_superchains_preserve_groups(model, sampler):
    if sampler == "nutpie":
        pytest.importorskip("nutpie")
    result = monitor.sample(model=model, draws=8, tune=8, chains=4, cores=1,
        until={"preset": "draft", "superchains": 2}, initial_strategy="prior",
        nuts_sampler=sampler, random_seed=4, progress=False)
    assert result.posterior.attrs[engine.cv.SUPERCHAIN_ATTR] == [0, 0, 1, 1]
    assert monitor.read_report(result)["checks"]["r_hat"]["nested"]


@pytest.mark.parametrize("fault", ["start", "inspect", "poll", "stop", "met"])
def test_nutpie_faults_never_disable_control(model, idata, monkeypatch, fault):
    from types import SimpleNamespace

    import nutpie

    control = RunControl()
    aborted = []
    calls = []
    class Handle:
        @property
        def is_finished(self):
            if fault == "poll":
                raise TimeoutError("poll failed")
            if fault == "stop":
                control.stop()
            return False
        def inspect(self):
            if fault == "inspect":
                raise RuntimeError("inspection failed")
            return idata
        def abort(self):
            aborted.append(True)
            return idata
    def sample(compiled_model, *, save_warmup=False, progress_callback=None, **kwargs):
        calls.append(kwargs)
        if fault == "start":
            raise RuntimeError("callback wiring failed")
        progress_callback([SimpleNamespace(tuning=False, finished_draws=0 if fault == "poll" else 200,
                                           divergences=0)] * 2)
        return Handle()
    monkeypatch.setattr(nutpie, "sample", sample)
    monkeypatch.setattr(nutpie, "compile_pymc_model", lambda *a, **kw: object())
    stream = engine.iter_sample(model, {"chains": 2, "tune": 5}, nuts_sampler="nutpie",
        until={"preset": "draft", "ess": 1, "r_hat": 2, "patience": 1}, control=control)
    if fault in ("stop", "met"):
        events, result = drain(stream)
        assert result.report["status"] == ("stopped" if fault == "stop" else "met")
        assert aborted
    else:
        with pytest.raises(RuntimeError, match={"start": "wiring", "inspect": "inspection",
                                               "poll": "polling failed"}[fault]):
            drain(stream)
        assert control.stopped
        if fault != "start":
            assert aborted
    assert len(calls) == 1


def test_nutpie_stop_is_latched_before_yield(model, idata, monkeypatch):
    from types import SimpleNamespace

    import nutpie

    aborted = []
    class Handle:
        is_finished = False
        def inspect(self):
            return idata
        def abort(self):
            aborted.append(True)
            return idata
    def sample(compiled, *, save_warmup=False, progress_callback=None, **kwargs):
        progress_callback([SimpleNamespace(tuning=False, finished_draws=200,
                                           divergences=0)] * 2)
        return Handle()
    monkeypatch.setattr(nutpie, "sample", sample)
    monkeypatch.setattr(nutpie, "compile_pymc_model", lambda *a, **kw: object())
    stream = engine.iter_sample(model, {"chains": 2, "tune": 5}, nuts_sampler="nutpie",
        until={"preset": "draft", "ess": 1, "r_hat": 2, "patience": 1})
    try:
        for event in stream:
            if isinstance(event, ConvergenceUpdate) and not event.final:
                assert aborted  # a slow consumer cannot postpone the abort
                break
    finally:
        stream.close()


def test_escalation_attempt_history(model, idata, monkeypatch):
    get_group(idata, "sample_stats")["diverging"].values[0, 0] = True
    settings = []
    def once(model, sampler, current, *args):
        settings.append(dict(current["extra"]))
        yield PhaseUpdate(engine.JobPhase.SAMPLING, "in_progress", "attempt", 0)
        return idata, "futile"
    monkeypatch.setattr(engine, "_once", once)
    _, result = drain(engine.iter_sample(model, {"chains": 2}, nuts_sampler="pymc",
                                         until={"escalate": 2}))
    assert len(result.report["attempts"]) == 3
    assert settings == [{}, {"target_accept": .95}, {"target_accept": .99}]
    assert monitor.read_report(result.idata) == result.report


def test_real_custom_native_step_and_zero_warmup():
    with pm.Model() as model:
        pm.Bernoulli("z", .5)
        step = pm.Metropolis()
    result = monitor.sample(model=model, step=step, draws=12, tune=0, chains=2,
                            cores=1, random_seed=2, progress=False)
    assert result.posterior.sizes["draw"] == 12
    report = monitor.read_report(result)
    assert not report["met"] and report["checks"]["bfmi"]["ok"] is False
