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
assert importlib.metadata.version("pymc-monitor") == monitor.__version__ == "0.1.0"
assert files("pymc_monitor").joinpath("static/renderer.js").is_file()
print("pymc-monitor smoke OK", report["status"], "PyMC", pm.__version__)
