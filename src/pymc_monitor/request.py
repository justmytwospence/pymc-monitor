"""Resolve sampler arguments once for execution and cache-key integration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import find_spec

from .control import RunControl
from .convergence import normalize_target
from .idata import pymc_major


def default_sampler(model, kwargs):
    """Prefer nutpie when installed and compatible; native steps stay native."""
    if kwargs.get("step") is not None or kwargs.get("callback") is not None:
        return "pymc"
    if any(str(rv.dtype).startswith(("int", "uint", "bool")) for rv in model.free_RVs):
        return "pymc"
    return "nutpie" if find_spec("nutpie") is not None else "pymc"


@dataclass
class SamplingRequest:
    model: object
    sampler: str
    target: dict | None
    names: list[str]
    ids: list[int] | None
    chains: int
    settings: dict
    initial_strategy: str | None
    control: RunControl


def resolve(model, sample_kwargs, *, nuts_sampler=None, until=None, initial_strategy=None,
            control=None) -> SamplingRequest:
    import pymc as pm

    if model is None:
        model = pm.modelcontext(None)
    kwargs = dict(sample_kwargs)
    if "progressbar" in kwargs:
        raise ValueError("progressbar is managed by pymc-monitor")
    if kwargs.get("return_inferencedata") is False:
        raise ValueError("return_inferencedata=False is incompatible with convergence reports")
    if "nuts_sampler" in kwargs or "until" in kwargs:
        raise ValueError("pass nuts_sampler and until as explicit arguments")
    if initial_strategy not in (None, "prior", "support_point"):
        raise ValueError("initial_strategy must be None, 'prior', or 'support_point'")
    sampler = nuts_sampler if nuts_sampler is not None else default_sampler(model, kwargs)
    if sampler not in ("pymc", "nutpie", "numpyro", "blackjax"):
        raise ValueError(f"unknown sampler {sampler!r}")
    if kwargs.get("step") is not None and sampler != "pymc":
        raise ValueError("custom step methods require nuts_sampler='pymc'")
    if kwargs.get("callback") is not None and (sampler != "pymc" or not callable(kwargs["callback"])):
        raise ValueError("callback must be callable and is only supported for the pymc sampler")
    if kwargs.get("callback") is None:
        kwargs.pop("callback", None)
    if kwargs.get("trace") is not None:
        raise ValueError("custom trace backends are not supported by monitored partial recovery")
    if initial_strategy is not None and kwargs.get("initvals") is not None:
        raise ValueError("initial_strategy and initvals cannot both be specified")
    target = normalize_target(until)
    if target and sampler in ("numpyro", "blackjax"):
        raise ValueError("JAX samplers cannot stream draws for adaptive stopping; choose pymc or nutpie")
    draws = kwargs.pop("draws", 1000)
    tune = kwargs.pop("tune", None)
    if tune is None:
        tune = 400 if sampler == "nutpie" and pymc_major() >= 6 else 1000
    chains = kwargs.pop("chains", None)
    cores = kwargs.get("cores")
    if cores is not None and (isinstance(cores, bool) or not isinstance(cores, int) or cores < 1):
        raise ValueError("cores must be a positive integer")
    if chains is None:
        chains = max(2, cores if cores is not None else min(4, os.cpu_count() or 4))
    for name, n, minimum in (("chains", chains, 1), ("draws", draws, 1), ("tune", tune, 0)):
        if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if sampler == "nutpie" and tune == 0:
        raise ValueError("nutpie requires tune >= 1; use the native sampler for zero warmup")
    ids = None
    superchains = (target or {}).get("superchains")
    if superchains:
        if kwargs.get("initvals") is not None or kwargs.get("step") is not None:
            raise ValueError("superchains manage initialization; initvals and step are incompatible")
        if chains % superchains or chains // superchains < 2:
            raise ValueError("superchains must divide chains into groups of at least two")
        ids = [c // (chains // superchains) for c in range(chains)]
    names = (target or {}).get("var_names") or [rv.name for rv in model.free_RVs]
    known = {rv.name for rv in model.free_RVs}
    output_names = kwargs.get("var_names")
    if output_names is not None and not set(names).issubset(output_names):
        raise ValueError("var_names must retain every monitored free variable")
    if not names or any(name not in known for name in names):
        raise ValueError(f"unknown or empty monitored free variables: {names!r}")
    if control is None:
        control = RunControl()
    if not isinstance(control, RunControl):
        raise TypeError("control must be a RunControl")
    return SamplingRequest(model, sampler, target, list(names), ids, chains,
                           {"draws": draws, "tune": tune,
                            "seed": kwargs.pop("random_seed", None), "extra": kwargs},
                           initial_strategy, control)
