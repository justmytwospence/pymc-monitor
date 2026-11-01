"""Sampler-independent local event stream and lifecycle control."""
from __future__ import annotations

import inspect
import queue
import threading
import time

import numpy as np

from . import convergence as cv
from .control import SamplingStoppedError
from .events import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    RunFinished,
    SamplingProgress,
    SamplingResult,
)
from .idata import concat_chains, get_group, to_inference_data, trim_incomplete_draws
from .request import resolve


def _phase(status, message, elapsed=0.0, phase=JobPhase.SAMPLING):
    return PhaseUpdate(phase, status, message, elapsed)


def _convergence(a, d, final=False):
    return ConvergenceUpdate(
        params={k: ParamConvergence(**v) for k, v in a.get("params", {}).items()},
        draws=d.n_draws,
        traces={name: d.values[:, ::max(1, (d.n_draws + 499) // 500), i].tolist()
                for i, name in enumerate(d.labels[:20])},
        verdict={k: a.get(k) for k in (
            "met", "decision", "consecutive", "projected_draws", "cause", "hint", "next_check", "checks"
        )}, final=final,
    )


def _nutpie_callback_supported(sample):
    """A catch-all **kwargs does not mean nutpie accepts this sampler setting."""
    try:
        param = inspect.signature(sample).parameters.get("progress_callback")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind in (
        inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _valid_prefix(group, chain):
    """Count real, finite draws for one chain, excluding a NaN-padded tail."""
    if group is None or "chain" not in group.sizes or "draw" not in group.sizes:
        return 0
    if chain >= group.sizes["chain"]:
        return 0
    n = int(group.sizes["draw"])
    if not n:
        return 0
    for name in group.data_vars:
        da = group[name]
        if "chain" not in da.dims or "draw" not in da.dims:
            continue
        vals = np.asarray(da.isel(chain=chain).transpose("draw", ...).values)
        if not np.issubdtype(vals.dtype, np.number):
            continue
        finite = np.isfinite(vals.reshape(vals.shape[0], -1)).all(axis=1)
        if not finite.all():
            n = min(n, int(np.argmax(~finite)))
    return n


def _nutpie_snapshot_progress(parts, jobs, chains, draws, tune, save_warmup, started):
    """Compute per-chain post-warmup counts from untrimmed snapshots."""
    progress = {}
    for part, (_, count, _, offset) in zip(parts, jobs):
        posterior = get_group(part, "posterior")
        warmup = get_group(part, "warmup_posterior") if save_warmup else None
        sample_stats = get_group(part, "sample_stats")
        for local in range(count):
            n = min(draws, _valid_prefix(posterior, local))
            w = min(tune, _valid_prefix(warmup, local)) if warmup is not None else 0
            if not n and warmup is not None and w < tune:
                phase, current, total = "tuning", w, tune
            elif n or (warmup is not None and w >= tune):
                phase, current, total = "sampling", n, draws
            else:
                phase, current, total = "indeterminate", 0, draws
            divergent = sample_stats.data_vars.get("diverging") if sample_stats is not None else None
            n_stats = min(n, _valid_prefix(sample_stats, local))
            num_div = (int(np.count_nonzero(divergent.isel(chain=local, draw=slice(0, n)).values))
                       if divergent is not None and n_stats >= n else None)
            depth = sample_stats.data_vars.get("depth") if sample_stats is not None else None
            step = sample_stats.data_vars.get("step_size") if sample_stats is not None else None
            steps = sample_stats.data_vars.get("n_steps") if sample_stats is not None else None
            mean_depth = (float(np.mean(depth.isel(chain=local, draw=slice(max(0, n - 100), n)).values))
                          if depth is not None and n_stats >= n and n else None)
            step_size = (float(step.isel(chain=local, draw=n - 1).values)
                         if step is not None and n_stats >= n and n else None)
            tree_size = (int(steps.isel(chain=local, draw=n - 1).values)
                         if steps is not None and n_stats >= n and n else None)
            elapsed = time.monotonic() - started
            speed = current / elapsed if current and elapsed > 0 else None
            eta = max(0, total - current) / speed if speed else None
            progress[offset + local] = ChainProgress(current, total, phase,
                draws_per_sec=speed, eta_seconds=eta, divergences=num_div,
                mean_tree_depth=mean_depth, step_size=step_size, tree_size=tree_size)
    # Missing jobs/chains are zero, not the padded draw dimension of a started chain.
    for i in range(chains):
        progress.setdefault(i, ChainProgress(0, draws, "indeterminate",
            draws_per_sec=None, eta_seconds=None, divergences=None,
            mean_tree_depth=None, step_size=None, tree_size=None))
    divs = [p.divergences for p in progress.values()]
    return SamplingProgress(progress, sum(divs) if all(d is not None for d in divs) else None,
                            time.monotonic() - started,
                            sum(p.draw for p in progress.values() if p.phase == "sampling"))


def _live_converter(model, names):
    """Evaluate free RVs from transformed value variables, never label a log-value as a constrained draw."""
    rvs = {rv.name: rv for rv in model.free_RVs}
    missing = set(names) - set(rvs)
    if missing:
        raise ValueError(f"unknown monitored free variables: {sorted(missing)}")
    outputs = model.replace_rvs_by_values([rvs[name] for name in names])
    inputs = list(model.value_vars)
    fn = model.compile_fn(outputs, inputs=inputs, point_fn=True, on_unused_input="ignore")
    return lambda point: dict(zip(names, fn({v.name: point[v.name] for v in inputs})))


def _assessment(idata, names, ids, target, controller, status, decision, attempts):
    post = get_group(idata, "posterior")
    if post is None or int(post.sizes.get("draw", 0)) == 0:
        raise SamplingStoppedError("sampling stopped before a complete posterior draw was available")
    if ids is not None:
        if int(post.sizes["chain"]) != len(ids):
            raise SamplingStoppedError("partial trace lost superchain membership")
        post.attrs[cv.SUPERCHAIN_ATTR] = list(ids)
    d = cv.draws_from_idata(idata, var_names=names, superchain_ids=ids)
    a = cv.assess(d, target)
    if not a["met"] and d.n_draws >= 4:
        a["cause"], a["hint"] = cv.diagnose(d, a)
    a["projected_draws"] = cv.projected_draws(a) if d.n_draws >= 4 else None
    if status == "met" and not a["met"]:
        status = "cap"  # the recovered posterior, not the checkpoint, is authoritative
    elif status == "futile" and a["met"]:
        status = "cap"
    a["decision"] = status if status != "complete" else "complete"
    report = cv.make_report(a, status=status, target=controller.target if controller.stop else None,
                            attempts=attempts or None)
    cv.attach_report(idata, report)
    return report, _convergence(a, d, final=True)


def _next_rung(report, settings, sampler, applied):
    from .policies import next_rung
    return next_rung(report, settings, sampler, applied)


def iter_sample(model, sample_kwargs, *, nuts_sampler=None, until=None, initial_strategy=None,
                control=None, sample_fn=None):
    """Yield typed events; return ``SamplingResult`` as ``StopIteration.value``.

    The generator owns a worker thread for blocking PyMC/JAX calls and polls nutpie
    handles in the calling thread. Closing the iterator requests a cooperative stop.
    """
    import pymc as pm

    if sample_fn is not None and not callable(sample_fn):
        raise TypeError("sample_fn must be callable")
    request = resolve(model, sample_kwargs, nuts_sampler=nuts_sampler, until=until,
                      initial_strategy=initial_strategy, control=control)
    model, sampler, target = request.model, request.sampler, request.target
    chains, ids, names = request.chains, request.ids, request.names
    control, settings = request.control, request.settings
    # Even display-only runs use the same final policy and a full report.
    policy = target or cv.normalize_target(True)
    attempts, applied = [], {}
    started = time.monotonic()
    completed = False
    stream = None
    try:
        if control.poll():
            raise SamplingStoppedError("sampling stopped before it started")
        yield _phase("in_progress", "sampling started")
        for attempt in range(1 + (target or {}).get("escalate", 0)):
            controller = cv.Controller(policy, chains, settings["draws"], stop=bool(target))
            stream = _once(model, sampler, settings, chains, names, ids, initial_strategy,
                           control, sample_fn or pm.sample, controller)
            try:
                while True:
                    yield next(stream)
            except StopIteration as end:
                idata, decision = end.value
            finally:
                stream.close()
            idata = trim_incomplete_draws(to_inference_data(idata))
            if control.stopped and sampler not in ("numpyro", "blackjax"):
                status = "stopped"
            elif decision in ("met", "futile"):
                status = decision
            else:
                status = "cap" if target else "complete"
            report, final_event = _assessment(idata, names, ids, policy, controller, status,
                                               decision, attempts)
            yield final_event
            status = report["status"]
            attempts.append({"attempt": attempt + 1, "status": status, "draws": settings["draws"],
                             "tune": settings["tune"], "cause": report.get("cause")})
            if target and status in ("futile", "cap") and not report["met"] and attempt < target["escalate"] and not control.stopped:
                rung = _next_rung(report, settings, sampler, applied)
                if rung is not None:
                    yield _phase("in_progress", f"re-running: {rung}", time.monotonic() - started)
                    continue
            if len(attempts) > 1:
                report["attempts"] = attempts
                cv.attach_report(idata, report)
            yield _phase("done", cv.verdict_message(report), time.monotonic() - started)
            completed = True
            yield RunFinished(status, report)
            return SamplingResult(idata, report, sampler)
    except Exception as exc:
        yield _phase("error", str(exc), time.monotonic() - started)
        yield RunFinished("error", message=str(exc))
        raise
    finally:
        if not completed:
            control.stop()
        if stream is not None:
            stream.close()


def _once(model, sampler, settings, chains, names, ids, init, control, sample_fn, controller):
    """One run; all transport and UI concerns are deliberately outside this function."""
    draws, tune = settings["draws"], settings["tune"]
    kw = dict(settings["extra"])
    seed = settings["seed"]
    events = queue.Queue(maxsize=1)
    publish_lock = threading.Lock()
    checkpoint = threading.Condition()

    def publish(event):
        # Progress is a snapshot, not a draw log. Bound memory for slow consumers.
        with publish_lock:
            try:
                events.get_nowait()
            except queue.Empty:
                pass
            events.put_nowait(event)

    state = {"idata": None, "error": None, "decision": None}
    traces = {}
    user_callback = kw.pop("callback", None)
    counts = {i: 0 for i in range(chains)}
    divergences = {i: 0 for i in range(chains)}
    values = {i: {name: [] for name in names} for i in range(chains)}
    stats = {i: {"diverging": [], "energy": [], "maxdepth": []} for i in range(chains)}
    timings = {i: time.monotonic() for i in range(chains)}
    chain_state = {}
    started = time.monotonic()
    converter = _live_converter(model, names) if sampler == "pymc" else None
    if init is not None and sampler != "nutpie":
        kw["initvals"] = {rv.name: init for rv in model.free_RVs}
    if ids and sampler == "pymc":
        from pymc.initial_point import make_initial_point_fn
        strategy = init or "support_point"
        fn = make_initial_point_fn(model=model, default_strategy=strategy,
                                   jitter_rvs=set(), return_transformed=False)
        rng = np.random.default_rng(seed)
        starts = [fn(int(s)) for s in rng.integers(0, 2**31 - 1, len(set(ids)))]
        kw["initvals"] = [starts[k] for k in ids]
        kw["init"] = str(kw.get("init", "adapt_diag")).replace("jitter+", "")

    def callback(trace, draw):
        chain = int(draw.chain)
        traces[chain] = trace
        if control.stopped or state["decision"] is not None:
            raise KeyboardInterrupt("pymc-monitor early stop")
        tuning = bool(draw.tuning)
        if tuning:
            current = int(getattr(draw, "draw_idx", 0)) + 1
        else:
            point = converter(draw.point)
            for name, val in point.items():
                values[chain][name].append(np.array(val, dtype=float, copy=True))
            s = draw.stats[0] if draw.stats else {}
            divergent = float(s.get("diverging", np.nan))
            stats[chain]["diverging"].append(divergent)
            divergences[chain] += int(np.isfinite(divergent) and divergent != 0)
            stats[chain]["energy"].append(float(s.get("energy", np.nan)))
            stats[chain]["maxdepth"].append(float(s.get("reached_max_treedepth", np.nan)))
            counts[chain] += 1
            current = counts[chain]
            if current == 1:
                timings[chain] = time.monotonic()
        elapsed = max(time.monotonic() - timings[chain], 1e-9)
        chain_state[chain] = ChainProgress(current, tune if tuning else draws,
            "tuning" if tuning else "sampling", current / elapsed,
            max(0, ((tune if tuning else draws) - current) * elapsed / max(current, 1)),
            divergences[chain],
            mean_tree_depth=float((draw.stats[0] if draw.stats else {}).get("depth", 0)),
            step_size=float((draw.stats[0] if draw.stats else {}).get("step_size", 0)),
            tree_size=int((draw.stats[0] if draw.stats else {}).get("tree_size", 0)))
        publish(SamplingProgress(dict(chain_state), sum(divergences.values()),
                                 time.monotonic() - started, sum(counts.values())))
        if user_callback is not None:
            try:
                user_callback(trace=trace, draw=draw)
            except KeyboardInterrupt:
                control.stop()
                raise
        # Fast samplers must not outrun a due adaptive checkpoint while the
        # event-loop thread computes diagnostics (or a consumer renders them).
        if controller.stop and not tuning:
            with checkpoint:
                while (controller.due(min(counts.values())) and not control.stopped
                       and state["decision"] is None):
                    checkpoint.wait(timeout=0.05)

    if sampler == "nutpie":
        import nutpie

        from .policies import split_nutpie_kwargs
        # PyMC common options are translated, never silently ignored.
        compile_kw = dict(kw.pop("compile_kwargs", {}) or {})
        if init is not None:
            compile_kw["default_initialization_strategy"] = init
        if "initvals" in kw:
            compile_kw["initial_points"] = kw.pop("initvals")
        if "var_names" in kw:
            compile_kw["var_names"] = kw.pop("var_names")
        kw.pop("return_inferencedata", None)
        kw.pop("compute_convergence_checks", None)  # our final report replaces PyMC's
        kw["save_warmup"] = not kw.pop("discard_tuned_samples", True)
        nested = kw.pop("nuts_sampler_kwargs", {}) or {}
        if set(kw) & set(nested):
            raise ValueError("duplicate nutpie settings in nuts_sampler_kwargs")
        kw.update(nested)
        if "max_treedepth" in kw:
            if "maxdepth" in kw:
                raise ValueError("specify only one of maxdepth and max_treedepth")
            kw["maxdepth"] = kw.pop("max_treedepth")
        forwarded, rejected = split_nutpie_kwargs(nutpie, kw)
        if rejected:
            raise ValueError(f"unsupported nutpie sample arguments: {sorted(rejected)}")
        yield _phase("in_progress", "compiling model with nutpie", phase=JobPhase.COMPILING)
        compiled = nutpie.compile_pymc_model(model, **compile_kw)
        yield _phase("done", "nutpie compilation complete", phase=JobPhase.COMPILING)
        if control.poll():
            raise SamplingStoppedError("sampling stopped during compilation")
        jobs = []
        if ids:
            from dataclasses import replace

            from .policies import fixed_point
            rng = np.random.default_rng(seed)
            seeds = [int(s) for s in rng.integers(0, 2**31 - 1, len(set(ids)) * 2)]
            for k in range(len(set(ids))):
                start = compiled.initial_point_func(seeds[k])
                jobs.append((replace(compiled, initial_point_func=fixed_point(start)),
                             chains // len(set(ids)), seeds[k + len(set(ids))], k * (chains // len(set(ids)))))
        else:
            jobs = [(compiled, chains, seed, 0)]
        callback_supported = _nutpie_callback_supported(nutpie.sample)
        if not callback_supported:
            yield _phase("in_progress", "nutpie progress via inspect() polling (no callback API)")
        handles = []
        parts = None
        post = {}
        def make_cb(offset):
            def on_progress(progress_list):
                for i, p in enumerate(progress_list):
                    c = offset + i
                    tuning = bool(p.tuning)
                    n = max(0, int(p.finished_draws or 0) - tune) if not tuning else int(p.finished_draws or 0)
                    post[c] = 0 if tuning else n
                    chain_state[c] = ChainProgress(n, tune if tuning else draws,
                        "tuning" if tuning else "sampling", divergences=int(p.divergences or 0))
                publish(SamplingProgress(dict(chain_state), sum(v.divergences for v in chain_state.values()),
                                         time.monotonic() - started, sum(post.values())))
            return on_progress
        try:
            for compiled_part, n, part_seed, offset in jobs:
                call = dict(draws=draws, tune=tune, chains=n, seed=part_seed,
                            blocking=False, progress_bar=False, **forwarded)
                if callback_supported:
                    call["progress_callback"] = make_cb(offset)
                handles.append(nutpie.sample(compiled_part, **call))
            failures = 0
            last_inspect = [-float("inf")] * len(handles)
            snapshots = [None] * len(handles)
            completed_parts = [None] * len(handles)
            progress = None
            while parts is None:
                try:
                    finished_handles = [h.is_finished for h in handles]
                    failures = 0
                except Exception as exc:
                    failures += 1
                    if failures > 10:
                        raise RuntimeError("nutpie polling failed repeatedly") from exc
                    finished_handles = [False] * len(handles)
                control.poll()
                if control.stopped:
                    parts = [trim_incomplete_draws(to_inference_data(h.abort())) for h in handles]
                    break
                if all(finished_handles):
                    parts = [completed_parts[i] if completed_parts[i] is not None else
                             to_inference_data(h.wait()) for i, h in enumerate(handles)]
                    break
                while not events.empty():
                    yield events.get_nowait()
                now = time.monotonic()
                refreshed = False
                due = controller.due(min(post.values()) if len(post) == chains else 0)
                for i, h in enumerate(handles):
                    if finished_handles[i]:
                        if completed_parts[i] is None:
                            completed_parts[i] = to_inference_data(h.wait())
                            refreshed = True
                        snapshots[i] = completed_parts[i]
                    elif (not callback_supported or due) and now - last_inspect[i] >= 0.5:
                        last_inspect[i] = now
                        snapshots[i] = to_inference_data(h.inspect())
                        refreshed = True
                if not callback_supported and refreshed and all(part is not None for part in snapshots):
                    progress = _nutpie_snapshot_progress(snapshots, jobs, chains, draws, tune,
                                                         forwarded["save_warmup"], started)
                    yield progress
                if not callback_supported:
                    n = (min(p.draw if p.phase == "sampling" else 0
                             for p in progress.chains.values()) if progress is not None else 0)
                    due = controller.due(n)
                if due and refreshed and not control.stopped and all(part is not None for part in snapshots):
                    partial = [trim_incomplete_draws(part) for part in snapshots]
                    d = cv.draws_from_idata(partial[0] if len(partial) == 1 else concat_chains(partial),
                                            var_names=names, superchain_ids=ids)
                    if controller.due(d.n_draws):
                        a = controller.step(d)
                        if a["decision"] in ("met", "futile"):
                            state["decision"] = a["decision"]
                            parts = [trim_incomplete_draws(to_inference_data(h.abort())) for h in handles]
                        yield _convergence(a, d)
                        if parts is not None:
                            break
                if control.stopped or state["decision"]:
                    parts = [trim_incomplete_draws(to_inference_data(h.abort())) for h in handles]
                    break
                time.sleep(0.1)
            while not events.empty():
                yield events.get_nowait()
            yield _nutpie_snapshot_progress(parts, jobs, chains, draws, tune,
                                            forwarded["save_warmup"], started)
            idata = parts[0] if len(parts) == 1 else concat_chains(parts)
            return idata, state["decision"]
        finally:
            if parts is None:
                for h in handles:
                    try:
                        h.abort()
                    except Exception:
                        pass

    if sampler in ("numpyro", "blackjax"):
        yield _phase("in_progress", f"{sampler} sampling (phase-only progress)")
    def run():
        try:
            with model:
                call = dict(draws=draws, tune=tune, chains=chains,
                            random_seed=seed, progressbar=False, nuts_sampler=sampler, **kw)
                if sampler == "pymc":
                    call["callback"] = callback
                state["idata"] = sample_fn(**call)
        except KeyboardInterrupt:
            # PyMC normally returns the partial idata itself on interrupt.
            pass
        except BaseException as exc:
            state["error"] = exc

    if control.poll():
        raise SamplingStoppedError("sampling stopped during initialization")
    thread = threading.Thread(target=run, daemon=True, name="pymc-monitor-sampler")
    thread.start()
    try:
        while thread.is_alive():
            control.poll()
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                event = None
            if event is not None:
                yield event
            if sampler == "pymc" and state["decision"] is None and controller.due(min(counts.values())) and not control.stopped:
                n = min(counts.values())
                try:
                    arrays = {name: np.stack([np.stack(values[c][name][:n]) for c in range(chains)])
                              for name in names}
                    d = cv.draws_from_arrays(arrays,
                        diverging=np.array([stats[c]["diverging"][:n] for c in range(chains)]),
                        energy=np.array([stats[c]["energy"][:n] for c in range(chains)]),
                        maxdepth=np.array([stats[c]["maxdepth"][:n] for c in range(chains)]),
                        superchain_ids=ids)
                    with checkpoint:
                        a = controller.step(d)
                        if a["decision"] in ("met", "futile"):
                            # Latch before resuming callbacks or yielding to UI.
                            state["decision"] = a["decision"]
                        checkpoint.notify_all()
                except Exception as exc:
                    control.stop()
                    raise RuntimeError("live diagnostics failed; adaptive safety cannot be guaranteed") from exc
                yield _convergence(a, d)
            if control.stopped and sampler in ("numpyro", "blackjax"):
                yield _phase("in_progress", "stop requested; JAX cannot abort early")
                # Only once; it must run to completion.
                while thread.is_alive():
                    thread.join(0.1)
    finally:
        if thread.is_alive() and sampler == "pymc":
            control.stop()
        thread.join()
    while not events.empty():
        yield events.get_nowait()
    if sampler == "pymc" and (control.stopped or state["decision"] is not None):
        if not min(counts.values()):
            if state["error"] is not None:
                raise SamplingStoppedError("no complete posterior draw across all chains") from state["error"]
            raise SamplingStoppedError("no complete posterior draw across all chains")
        if state["error"] is None:
            import pymc as pm
            from pymc.backends.base import MultiTrace

            n = min(counts.values())
            recovered = MultiTrace([traces[c][tune:tune + n] for c in range(chains)])
            state["idata"] = pm.to_inference_data(recovered, model=model,
                                                 **(kw.get("idata_kwargs") or {}))
    if state["error"] is not None:
        raise state["error"]
    if state["idata"] is None:
        raise SamplingStoppedError("sampler did not return a recoverable partial posterior")
    return state["idata"], state["decision"]
