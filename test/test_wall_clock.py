"""Guard: the sim stack's nodes keep time with ROS time, so sim_world's simulated clock (~speed) drives them.

A wall-clock call (time.time / time.sleep / time.monotonic) in these files must carry a "# wall clock:"
comment on the same line saying why (compute budgets, CPU throttles, pacing the sim clock itself).
Anything the robot reasons about (windows, cooldowns, deadlines, stamps, belief decay) must use
rospy.get_time() / rospy.Time.now(), or a sped-up run silently mixes two clocks. No ROS needed.
"""

import os
import re

import pytest

SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NODES = [
    "mattbot_sim/scripts/sim_world.py",
    "mattbot_sim/scripts/scenario_runner.py",
    "mattbot_sim/scripts/sim_monitor.py",
    "mattbot_sim/src/mattbot_sim/clock.py",
    "mattbot_navigation/scripts/localize_and_navigate.py",
    "mattbot_navigation/scripts/observation_planner.py",
    "mattbot_navigation/scripts/occupancy_grid_mapper.py",
    "mattbot_navigation/src/navigation_utils/search.py",
    "mattbot_navigation/src/navigation_utils/obstacle_importance.py",
    "mattbot_dds/scripts/observation_ledger.py",
    "mattbot_dds/scripts/object_belief_map.py",
    "mattbot_image_detection/scripts/observation_evaluator.py",
]
WALL = re.compile(r"\btime\.(time|sleep|monotonic|perf_counter)\(")


@pytest.mark.parametrize("rel", NODES)
def test_wall_clock_calls_are_marked(rel):
    path = os.path.join(SRC, rel)
    if not os.path.exists(path):
        pytest.skip("%s not in this workspace" % rel)
    unmarked = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            code = line.split("#", 1)[0]
            if WALL.search(code) and "# wall clock:" not in line:
                unmarked.append("%s:%d: %s" % (rel, n, line.strip()))
    assert not unmarked, "wall-clock calls without a '# wall clock:' reason:\n" + "\n".join(unmarked)
