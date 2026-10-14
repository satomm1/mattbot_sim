"""The shipped scout scenarios really test what they claim (offline, mattbot_sim/detour_check.py check_scout).

Objects annotated ``scout: expected`` must get a scout from some pose where the robot sits idle long enough,
``scout: declined`` ones from none, and the sim camera must not see an annotated object from an idle pose.
Settings come from each scenario's launch_args. Needs mattbot_navigation's navigation_utils and path_planning's
social_path_planning; skipped otherwise.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub in ("mattbot_navigation/src", "path_planning/src"):
    sys.path.insert(0, os.path.join(SRC, sub))
pytest.importorskip("navigation_utils.scout")

from mattbot_sim.detour_check import check_scout, idle_poses, scout_failures  # noqa: E402
from mattbot_sim.scenario import load_scenario  # noqa: E402

SCENARIOS = os.path.join(HERE, "..", "scenarios")
MAP_DIR = os.path.join(SRC, "mattbot_mcl", "map_json")
ANNOTATED = sorted(
    f for f in os.listdir(SCENARIOS)
    if f.endswith(".yaml") and load_scenario(os.path.join(SCENARIOS, f)).scout
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("roadmap"))


def test_some_scenarios_are_annotated():
    assert {"scout_present.yaml", "scout_not_worth_it.yaml", "scout_expensive.yaml"} <= set(ANNOTATED)


@pytest.mark.skipif(not os.path.isdir(MAP_DIR), reason="mattbot_mcl/map_json not found")
@pytest.mark.parametrize("name", ANNOTATED)
def test_scout_annotations(name, cache_dir):
    sc = load_scenario(os.path.join(SCENARIOS, name))
    results = check_scout(sc, MAP_DIR, cache_dir, object_ids=set(sc.scout))
    assert set(r.object_id for r in results) == set(sc.scout)
    assert all(oid in sc.known_from_peer for oid in sc.scout)  # hidden objects must reach the ledger somehow
    assert scout_failures(sc, results, MAP_DIR) == []


def test_idle_poses_follow_the_runner():
    patrol = load_scenario(os.path.join(SCENARIOS, "scout_patrol_busy.yaml"))
    assert idle_poses(patrol, 10.0) == []  # 1.5 s settle + 2 s pause per waypoint
    assert len(idle_poses(patrol, 3.0)) == 2
    present = load_scenario(os.path.join(SCENARIOS, "scout_present.yaml"))
    assert [why for _p, why in idle_poses(present, 10.0)] == ["end of patrol"]
