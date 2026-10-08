"""The fast simulator against the ROS simulator on the same scenarios (fixtures/ros: ROS run summaries).

Counts must agree within tolerance: checks done (per stop kind, within max(1, 15 %)), ABSENT outcomes, correct /
false removals and ledger objects exactly, removal latency within 25 %. If this fails after a change to the
robot code, update the fast navigator model (or re-record the fixture with the ROS sim), not the tolerances.
"""

import glob
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
pytest.importorskip("mattbot_sim.fast.imports")
from mattbot_sim.fast import imports  # noqa: E402

pytest.importorskip("navigation_utils")
if not os.path.isdir(imports.MAP_JSON_DIR):
    pytest.skip("mattbot_mcl/map_json not found", allow_module_level=True)
from mattbot_sim.fast.sim import FastSim  # noqa: E402
from mattbot_sim.scoring import summary_value  # noqa: E402

FIXTURES = sorted(glob.glob(os.path.join(HERE, "fixtures", "ros", "*.json")))
EXACT = ["ledger_objects_added", "correct_removals", "false_removals", "observation_outcomes.ABSENT",
         "stale_ledger_objects"]
COUNTS = ["observation_stops_by_kind.OPPORTUNISTIC.ENDED", "observation_stops_by_kind.DETOUR.ENDED"]


@pytest.fixture(scope="module", params=FIXTURES, ids=lambda p: os.path.basename(p)[:-5])
def pair(request):
    with open(request.param) as f:
        fixture = json.load(f)
    path = os.path.join(HERE, "..", "scenarios", fixture["scenario"] + ".yaml")
    fast = FastSim(path, seed=0).run()
    return fixture["summaries"], fast


def test_counts_agree(pair):
    ros, fast = pair
    for rs in ros:
        fs = fast[rs.get("robot_id", next(iter(fast)))]
        for field in EXACT:
            assert summary_value(fs, field) == summary_value(rs, field), field
        for field in COUNTS:
            r, f = summary_value(rs, field), summary_value(fs, field)
            assert abs(f - r) <= max(1, 0.15 * r), (field, r, f)
        rl, fl = rs.get("removal_latency_s"), fs.get("removal_latency_s")
        assert (rl is None) == (fl is None)
        if rl:
            assert fl["mean"] == pytest.approx(rl["mean"], rel=0.25)
