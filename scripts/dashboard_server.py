"""Browser fixture: two isolated runs on the same dashboard with hostile labels."""
import json
import time

from pymc_monitor.dashboard import Dashboard, DashboardState
from pymc_monitor.events import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    SamplingProgress,
)
from pymc_monitor.sampling import RunControl

control = RunControl()
state = DashboardState()
other = DashboardState()
service = Dashboard()
run_id = service.add_run(state, control)
other_id = service.add_run(other, RunControl())
state.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "working <img src=x onerror=window.pwned=1>", 0))
state.show_sampling(SamplingProgress({0: ChainProgress(50, 100, "sampling")}))
state.show_convergence(ConvergenceUpdate(
    {"<script>window.pwned=1</script>": ParamConvergence(1.02, 80, 60)}, draws=50,
    traces={"model-a/x": [[1, 2, 1], [2, 1, 2]], "model-b/y": [[2, 3, 2]]},
    verdict={"decision": "continue", "checks": {"ess_bulk": {"value": 80, "threshold": 100, "ok": False}}},
))
print(json.dumps([service.url(run_id), service.url(other_id)]), flush=True)
try:
    while True:
        time.sleep(.1)
        if control.stopped and not state.snapshot()["complete"]:
            state.show_phase(PhaseUpdate(JobPhase.SAMPLING, "done", "stopped by user", 1))
            state.finish("stopped")
finally:
    service.close()
