"""Execute against an installed wheel or sdist in an isolated environment."""
import importlib.metadata
from importlib.resources import files

import pymc as pm

import pymc_monitor as monitor

with pm.Model() as model:
    pm.Normal("x")
result = monitor.sample(model=model, draws=5, tune=5, chains=2, cores=1,
                        nuts_sampler="pymc", random_seed=42, progress=False)
report = monitor.read_report(result)
assert report is not None and report["draws"] == 5
assert callable(monitor.sample.__pymc_cache_normalize__)
normalized = monitor.sample.__pymc_cache_normalize__(model, {"draws": 5, "nuts_sampler": "pymc"})
assert set(normalized) == {"call_kwargs", "key_kwargs", "sampler_identity", "versions"}
assert "model" not in normalized["call_kwargs"]
assert result.posterior.sizes["draw"] == 5
assert importlib.metadata.version("pymc-monitor") == monitor.__version__ == "0.1.0"
assert files("pymc_monitor").joinpath("static/uPlot.iife.min.js").is_file()
assert "<script>" in monitor.render_dashboard_html(progress_url="/progress", stop_url=None,
                                                   token="")
print("pymc-monitor smoke OK", report["status"], "PyMC", pm.__version__)
