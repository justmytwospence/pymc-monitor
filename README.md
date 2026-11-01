# pymc-monitor

Local streaming convergence diagnostics and adaptive stopping for PyMC 5.28 and 6.

```python
import pymc as pm
import pymc_monitor as monitor

with pm.Model() as model:
    mu = pm.Normal("mu")
    y = pm.Normal("y", mu, observed=[0.1, -0.2])

idata = monitor.sample(model=model, draws=1000, tune=500, until="draft")
print(monitor.read_report(idata))
```

`sample` returns the inference data. `start` returns a `SamplingHandle` with `wait()`, `stop()`, `close()`, `done`, `snapshot()` and `dashboard_url`. For custom consumers, `iter_sample(model, sample_kwargs, ...)` yields typed `PhaseUpdate`, `SamplingProgress`, and `ConvergenceUpdate` events; its `StopIteration.value` is a `SamplingResult` containing `idata` and `report`. `RunControl(poll=predicate)` may be shared between a caller and either API; poll errors stop the run and propagate. JAX samplers have phase-only progress and cannot abort early; requesting adaptive stopping with JAX raises `ValueError`.

The default sampler is nutpie when installed for a continuous model, otherwise native PyMC. Custom steps, discrete free variables and native callbacks select PyMC. Set `nuts_sampler` explicitly for reproducibility.

Install `pymc-monitor[notebook]` for notebook widgets, `[nutpie]` for background nutpie sampling, or `[jax]` for JAX samplers. `dashboard=True` starts an authenticated local loopback dashboard. No Modal or cloud cache is required.

`until=True` uses the `final` policy; `until="draft"` uses development thresholds; a dict can override target and controller settings. Diagnostics use every element of every monitored free variable and HMC sample statistics. `check_convergence(idata, until=True)` independently assesses a completed trace. Stopped runs include a final report when at least one complete post-warmup draw can be recovered.

See the [quick start](docs/index.qmd), [sampler compatibility and stop lifecycle](docs/samplers.qmd), [convergence policies](docs/convergence.qmd), [dashboard guide](docs/dashboard.qmd), [executable examples](docs/examples.qmd), and [API reference](docs/api.qmd). Use `start()` with a context manager when you want to keep a dashboard open after sampling; `sample()` closes its resources before returning.
