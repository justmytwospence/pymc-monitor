"""Progress tracking data structures and PyMC callback factory."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Iterator

from .convergence import _jsonable


class JobPhase(str, Enum):
    SERIALIZING = "serializing"
    CACHE_HIT = "cache_hit"
    DATA_UPLOADING = "data_uploading"
    UPLOADING = "uploading"
    PROVISIONING = "provisioning"
    CONTAINER_READY = "container_ready"
    DEVICE = "device"
    COMPILING = "compiling"
    SAMPLING = "sampling"
    DOWNLOADING = "downloading"


@dataclass
class PhaseUpdate:
    phase: JobPhase
    status: str  # "in_progress", "done", "error"
    message: str
    elapsed: float
    progress: float | None = None  # 0-1 fraction, None if indeterminate


@dataclass
class ChainProgress:
    draw: int
    total: int
    phase: str  # "indeterminate" | "tuning" | "sampling" | "done"
    draws_per_sec: float | None = 0.0
    eta_seconds: float | None = 0.0
    divergences: int | None = 0
    mean_tree_depth: float | None = 0.0
    step_size: float | None = 0.0
    tree_size: int | None = 0  # grad evals (leapfrog steps)


@dataclass
class SamplingProgress:
    chains: dict[int, ChainProgress]
    total_divergences: int | None = 0
    elapsed: float = 0.0
    total_draws: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParamConvergence:
    rhat: float | None
    ess_bulk: int
    ess_tail: int


@dataclass
class ConvergenceUpdate:
    params: dict[str, ParamConvergence]
    draws: int = 0
    traces: dict[str, list[list[float]]] = field(default_factory=dict)  # param -> [[chain0_vals], [chain1_vals], ...]
    # The convergence layer's verdict at this checkpoint: met, decision
    # (continue/met/futile/complete), per-check value/threshold/ok/worst,
    # projected_draws, cause and hint. None from older workers.
    verdict: dict | None = None
    final: bool = False


@dataclass
class SamplingResult:
    """Generator return value: recovered inference data and its final report."""

    idata: object
    report: dict
    effective_sampler: str | None = None


@dataclass
class RunFinished:
    status: str
    report: dict | None = None
    message: str = ""


ProgressEvent = PhaseUpdate | SamplingProgress | ConvergenceUpdate | RunFinished


def event_to_dict(event: ProgressEvent) -> dict:
    """Encode a progress event for JSON/msgpack cloud transports (wire version 1)."""
    kinds = {PhaseUpdate: "phase", SamplingProgress: "sampling",
             ConvergenceUpdate: "convergence", RunFinished: "run_finished"}
    kind = kinds.get(type(event))
    if kind is None:
        raise TypeError(f"unsupported event: {type(event).__name__}")
    data = asdict(event)
    if isinstance(event, PhaseUpdate):
        data["phase"] = event.phase.value
    return {"schema_version": 1, "type": kind, **_jsonable(data)}


def event_from_dict(data: dict) -> ProgressEvent | None:
    """Decode both version-1 and legacy cloud messages; skip unknown types."""
    return decode_progress_event(data)


def _as_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _optional_float(value) -> float | None:
    result = _as_float(value, float("nan"))
    return result if math.isfinite(result) else None


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def decode_progress_event(data: dict) -> ProgressEvent | None:
    """Convert a decoded msgpack dict into a typed ProgressEvent.

    Shared by the client streaming path and the cp.map worker (which writes
    progress server-side), so it lives here with no Modal/backend dependency.

    Every field is read defensively and numerics are coerced: a partial or
    stringly payload from a newer/older worker previously raised KeyError
    mid-decode, and a null slipping through reached the dashboard where
    ``.toFixed`` on it read as the run having gone offline.
    """
    if not isinstance(data, dict) or data.get("schema_version", 1) != 1:
        return None
    msg_type = data.get("type")

    if msg_type == "phase":
        # Tolerate phase names this client doesn't know (a newer worker may add
        # one); skipping the event is better than the caller mistaking the
        # undecodable chunk for the result payload.
        try:
            phase = JobPhase(data["phase"])
        except (KeyError, ValueError):
            return None
        return PhaseUpdate(
            phase=phase,
            status=data.get("status", "in_progress"),
            message=data.get("message", ""),
            elapsed=_as_float(data.get("elapsed")),
            progress=_optional_float(data.get("progress")),
        )

    if msg_type == "sampling":
        chains = {}
        raw_chains = data.get("chains")
        for chain_id_str, cdata in (raw_chains if isinstance(raw_chains, dict) else {}).items():
            try:
                chain_id = int(chain_id_str)
                if not isinstance(cdata, dict):
                    continue
            except (TypeError, ValueError):
                continue
            chains[chain_id] = ChainProgress(
                draw=_as_int(cdata.get("draw")),
                total=_as_int(cdata.get("total")),
                phase=cdata.get("phase", "sampling"),
                draws_per_sec=(_optional_float(cdata["draws_per_sec"])
                               if "draws_per_sec" in cdata else 0.0),
                eta_seconds=(_optional_float(cdata["eta_seconds"])
                             if "eta_seconds" in cdata else 0.0),
                divergences=(_as_int(cdata["divergences"])
                             if cdata.get("divergences") is not None else
                             None if "divergences" in cdata else 0),
                mean_tree_depth=(_optional_float(cdata["mean_tree_depth"])
                                 if "mean_tree_depth" in cdata else 0.0),
                step_size=(_optional_float(cdata["step_size"])
                           if "step_size" in cdata else 0.0),
                tree_size=(_as_int(cdata["tree_size"])
                           if cdata.get("tree_size") is not None else
                           None if "tree_size" in cdata else 0),
            )
        return SamplingProgress(
            chains=chains,
            total_divergences=(_as_int(data["total_divergences"])
                               if data.get("total_divergences") is not None else
                               None if "total_divergences" in data else 0),
            elapsed=_as_float(data.get("elapsed")),
            total_draws=_as_int(data.get("total_draws")),
            warnings=data.get("warnings") if isinstance(data.get("warnings"), list) else [],
        )

    if msg_type == "convergence":
        params = {}
        raw_params = data.get("params")
        for name, pdata in (raw_params if isinstance(raw_params, dict) else {}).items():
            if not isinstance(pdata, dict):
                continue
            params[str(name)] = ParamConvergence(
                rhat=_optional_float(pdata.get("rhat")),
                ess_bulk=_as_float(pdata.get("ess_bulk")),
                ess_tail=_as_float(pdata.get("ess_tail")),
            )
        traces = data.get("traces")
        traces = _jsonable(traces) if isinstance(traces, dict) else {}
        verdict = data.get("verdict")
        return ConvergenceUpdate(
            params=params, draws=_as_int(data.get("draws")), traces=traces,
            verdict=verdict if isinstance(verdict, dict) else None,
            final=bool(data.get("final", data.get("partial") is False)),
        )

    if msg_type == "run_finished":
        report = data.get("report")
        return RunFinished(str(data.get("status", "complete")),
                           report if isinstance(report, dict) else None,
                           str(data.get("message", "")))

    return None


def dispatch_event(event: ProgressEvent | None, sinks) -> None:
    """Route a ProgressEvent to the matching method on each sink.

    Single source of truth for the event->sink fan-out used by both the
    client display loop (api.py) and the cp.map worker's dashboard writer.
    """
    if event is None:
        return
    for sink in sinks:
        if isinstance(event, PhaseUpdate):
            sink.show_phase(event)
        elif isinstance(event, SamplingProgress):
            sink.show_sampling(event)
        elif isinstance(event, ConvergenceUpdate) and hasattr(sink, "show_convergence"):
            sink.show_convergence(event)
        elif isinstance(event, RunFinished) and hasattr(sink, "finish"):
            sink.finish(event.status)


def make_sampling_callback(queue: Queue, tune: int, draws: int):
    """Create a PyMC sampling callback that pushes progress to a queue.

    The callback receives (trace, draw) on each MCMC iteration.
    PyMC's draw object provides: chain, tuning, stats, etc.

    NOTE: this mirrors the per-draw bookkeeping in the remote worker's
    ``progress_callback`` (pymc_monitor/remote/worker.py) -- the worker
    version additionally accumulates traces and honors the stop flag. Keep the
    draw counting / phase reset / windowed tree-depth logic in sync.
    """
    chain_draw_counts: dict[int, int] = {}
    chain_start_times: dict[int, float] = {}
    chain_divergences: dict[int, int] = {}
    chain_tree_depths: dict[int, list[float]] = {}
    chain_phase: dict[int, bool] = {}  # chain -> is_tuning

    def callback(trace, draw):
        chain = draw.chain
        is_tuning = draw.tuning

        if chain not in chain_start_times:
            chain_start_times[chain] = time.time()
            chain_draw_counts[chain] = 0
            chain_divergences[chain] = 0
            chain_tree_depths[chain] = []
            chain_phase[chain] = is_tuning

        # Tuning -> sampling transition: restart the draw count and clock so
        # progress shows draws out of `draws` (not tune+draws) with a sane ETA.
        if chain_phase.get(chain) and not is_tuning:
            chain_draw_counts[chain] = 0
            chain_start_times[chain] = time.time()
            chain_phase[chain] = False

        chain_draw_counts[chain] += 1
        current_draw = chain_draw_counts[chain]

        stats = draw.stats[0] if draw.stats else {}
        diverging = stats.get("diverging", False)
        tree_depth = stats.get("tree_depth", 0)
        tree_size = stats.get("n_steps", stats.get("tree_size", 0))
        step_size = stats.get("step_size", 0.0)

        if diverging:
            chain_divergences[chain] += 1
        chain_tree_depths[chain].append(tree_depth)

        elapsed = time.time() - chain_start_times[chain]
        dps = current_draw / elapsed if elapsed > 0 else 0.0
        total = tune if is_tuning else draws
        remaining = max(0, total - current_draw)
        eta = remaining / dps if dps > 0 else 0.0

        recent_td = chain_tree_depths[chain][-100:]
        mean_td = sum(recent_td) / len(recent_td)

        progress = ChainProgress(
            draw=current_draw,
            total=total,
            phase="tuning" if is_tuning else "sampling",
            draws_per_sec=dps,
            eta_seconds=eta,
            divergences=chain_divergences[chain],
            mean_tree_depth=mean_td,
            step_size=step_size,
            tree_size=tree_size,
        )
        queue.put((chain, progress))

    return callback


class ProgressAggregator:
    """Reads per-draw events from queue, emits batched SamplingProgress snapshots."""

    def __init__(self, queue: Queue, interval: float = 0.5):
        self._queue = queue
        self._interval = interval
        self._chains: dict[int, ChainProgress] = {}
        self._start_time = time.time()
        self._stopped = False

    def snapshots(self) -> Iterator[SamplingProgress]:
        """Yield aggregated snapshots at regular intervals."""
        while not self._stopped:
            deadline = time.time() + self._interval
            # Drain the queue until the interval elapses. `continue` on an
            # empty queue, not `break`: breaking out on the first timeout made
            # an idle chain re-emit an identical snapshot every ~0.1s instead
            # of once per interval -- ~5x the intended widget re-renders.
            while time.time() < deadline and not self._stopped:
                try:
                    chain, progress = self._queue.get(timeout=0.1)
                    self._chains[chain] = progress
                except Empty:
                    continue
                except Exception:
                    break

            snapshot = self._snapshot()
            if snapshot is not None:
                yield snapshot

        # Drain the final queued updates too, not just the previous snapshot.
        while True:
            try:
                chain, progress = self._queue.get_nowait()
            except Empty:
                break
            self._chains[chain] = progress
        final = self._snapshot()
        if final is not None:
            yield final

    def _snapshot(self) -> SamplingProgress | None:
        if not self._chains:
            return None
        total_div = (sum(c.divergences for c in self._chains.values())
                     if all(c.divergences is not None for c in self._chains.values()) else None)
        warnings = []
        if total_div is not None and total_div > 0:
            warnings.append(f"{total_div} divergence(s) so far")
        return SamplingProgress(
            chains=dict(self._chains),
            total_divergences=total_div,
            elapsed=time.time() - self._start_time,
            warnings=warnings,
        )

    def stop(self):
        self._stopped = True
