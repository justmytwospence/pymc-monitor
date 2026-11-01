"""Browser test fixture: start the real dashboard with deterministic events."""
import time

from pymc_monitor.dashboard import DashboardService
from pymc_monitor.progress import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    SamplingProgress,
)
from pymc_monitor.sampling import RunControl

control = RunControl()
service = DashboardService(control)
service.start()
service.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "working <img src=x onerror=window.pwned=1>", 0))
service.show_sampling(SamplingProgress({0: ChainProgress(50, 100, "sampling")}))
service.show_convergence(ConvergenceUpdate(
    {"<script>window.pwned=1</script>": ParamConvergence(1.02, 80, 60)}, draws=50,
    traces={"x": [[1, 2, 1], [2, 1, 2]]},
    verdict={"decision": "continue", "checks": {"ess_bulk": {"value": 80, "threshold": 100, "ok": False}}},
))
print(service.url, flush=True)
try:
    while True:
        time.sleep(1)
        if control.stopped:
            service.show_phase(PhaseUpdate(JobPhase.SAMPLING, "done", "stopped by user", 1))
            service.finish()
finally:
    service.close()
