"""Wire protocol stays compatible with partial and legacy cloud event streams."""
import json

from pymc_monitor.events import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    RunFinished,
    SamplingProgress,
    event_from_dict,
    event_to_dict,
)


def test_versioned_roundtrip_and_nonfinite():
    events = [PhaseUpdate(JobPhase.SAMPLING, "done", "done", 1.0),
              SamplingProgress({0: ChainProgress(1, 4, "sampling", eta_seconds=float("inf"))}),
              ConvergenceUpdate({"x": ParamConvergence(float("nan"), 4, 2)},
                                traces={"x": [[float("nan")]]}, final=True),
              RunFinished("met", {"status": "met"})]
    for event in events:
        encoded = event_to_dict(event)
        assert encoded["schema_version"] == 1
        json.dumps(encoded, allow_nan=False)
        decoded = event_from_dict(encoded)
        assert type(decoded) is type(event)
    assert event_to_dict(events[1])["chains"]["0"]["eta_seconds"] is None
    assert event_to_dict(events[2])["traces"]["x"][0][0] is None
    assert event_from_dict(event_to_dict(events[2])).final


def test_unknown_versions_and_malformed_entries_are_skipped():
    assert event_from_dict({"schema_version": 2, "type": "sampling"}) is None
    assert event_from_dict({"type": "future"}) is None
    assert event_from_dict({"type": "sampling", "chains": {"bad": {}, "0": None,
                            "1": {"draw": "2"}}}).chains[1].draw == 2
    assert not event_from_dict({"type": "convergence", "params": {"x": None}}).params
    assert event_from_dict({"type": "phase", "phase": "future"}) is None
