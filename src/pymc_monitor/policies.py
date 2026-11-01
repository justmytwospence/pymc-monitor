"""Retry ladder and nutpie argument compatibility (no transport dependencies)."""
import inspect
import math

LADDER = {
    "early_drift": [("tune", 2), ("tune", 2)],
    "divergences": [("target_accept", 0.95), ("target_accept", 0.99)],
    "low_bfmi": [("tune", 2), ("adaptation", "low_rank")],
    "chains_disagree": [("tune", 2), ("target_accept", 0.95)],
    "slow_mixing": [("draws", "projected")],
    "ess_stalled": [("adaptation", "low_rank"), ("tune", 2)],
}


def next_rung(report, settings, sampler, applied):
    cause = report.get("cause")
    steps = LADDER.get(cause, [])
    i = applied.get(cause, 0)
    while i < len(steps):
        key, value = steps[i]
        i += 1
        applied[cause] = i
        if key == "tune":
            old = settings["tune"]
            settings["tune"] = max(old + 1, math.ceil(old * value))
            return f"tune {old} -> {settings['tune']}"
        if key == "target_accept":
            old = settings["extra"].get(key, 0.8)
            if old >= value:
                continue
            settings["extra"][key] = value
            return f"target_accept {old} -> {value}"
        if key == "adaptation":
            if sampler != "nutpie":
                continue
            settings["extra"][key] = value
            settings["tune"] = max(800, settings["tune"])
            return f"adaptation -> {value}"
        if key == "draws":
            old = settings["draws"]
            projected = report.get("projected_draws")
            if not projected or projected <= old:
                continue
            settings["draws"] = min(4 * old, math.ceil(projected * 1.2))
            return f"draws {old} -> {settings['draws']}"
    return None


def fixed_point(start):
    def point(_seed):
        return start.copy()
    return point


def split_nutpie_kwargs(nutpie, kwargs):
    sig = inspect.signature(nutpie.sample)
    supported = set(sig.parameters) - {"kwargs", "compiled_model"}
    # Sampler settings accepted through nutpie's **kwargs, not its signature.
    supported |= {"target_accept", "maxdepth", "max_energy_error", "store_divergences",
                  "store_gradient", "store_mass_matrix", "use_grad_based_mass_matrix"}
    reserved = {"draws", "tune", "chains", "seed", "blocking", "progress_bar",
                "progress_callback", "return_raw_trace", "sampler"}
    supported -= reserved
    forwarded = {k: v for k, v in kwargs.items() if k in supported}
    return forwarded, {k: v for k, v in kwargs.items() if k not in supported}
