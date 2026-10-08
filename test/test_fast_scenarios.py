"""Every shipped scenario runs in the fast simulator, and its expectations pass (no ROS needed).

Slow-ish (a few seconds per scenario); runs in parallel-free order. Needs the workspace's robot libraries
(navigation_utils, dds_utils, observation_eval) and mattbot_mcl's maps; skipped otherwise.
"""

import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(HERE, "..", "scenarios")
pytest.importorskip("mattbot_sim.fast.imports")
from mattbot_sim.fast import imports  # noqa: E402

pytest.importorskip("navigation_utils")
if not os.path.isdir(imports.MAP_JSON_DIR):
    pytest.skip("mattbot_mcl/map_json not found", allow_module_level=True)
from mattbot_sim.fast.sim import FastSim  # noqa: E402

NAMES = sorted(f[:-5] for f in os.listdir(SCENARIOS) if f.endswith(".yaml"))


@pytest.mark.parametrize("name", NAMES)
def test_scenario_runs_and_passes(name):
    sim = FastSim(os.path.join(SCENARIOS, name + ".yaml"), seed=0)
    summaries = sim.run()
    assert summaries and all(s["run_s"] >= sim.duration - 1.0 for s in summaries.values())
    failed = [(rid, c) for rid, s in summaries.items() for c in s.get("expectations", []) if not c["ok"]]
    assert not failed, failed
    assert all(s["false_removals"] == 0 for s in summaries.values())


def test_same_seed_same_result():
    path = os.path.join(SCENARIOS, "remove_one.yaml")
    a = FastSim(path, seed=3, duration=250).run()
    b = FastSim(path, seed=3, duration=250).run()
    for s in list(a.values()) + list(b.values()):
        s.pop("run_wall_s")
    assert a == b


def test_robots_pass_each_other_head_on(tmp_path):
    path = tmp_path / "headon.yaml"
    path.write_text(
        "name: headon\nmap: current_map\nduration: 120\nloop: false\nobjects: []\nrobots:\n"
        "  - {id: 51, start: {x: 30.0, y: 18.95, theta: 0.0}, waypoints: [{x: 44.0, y: 18.95, theta: 0.0}]}\n"
        "  - {id: 52, start: {x: 44.0, y: 18.95, theta: 3.14159}, waypoints: [{x: 30.0, y: 18.95, theta: 3.14159}]}\n")
    import math

    sim = FastSim(str(path))
    sim.orchestrator.on_start(sim)
    closest = math.inf
    for _ in range(int(sim.duration / sim.dt)):
        sim.step()
        a, b = sim.robots[51].pose, sim.robots[52].pose
        closest = min(closest, math.hypot(a.x - b.x, a.y - b.y))
    assert closest >= 2 * sim.params.robot_radius  # never drove through each other
    assert sim.robots[51].pose.x > 43.5 and sim.robots[52].pose.x < 30.5  # both got past
    assert sim.robots[51].nav.counters["robot_replans"] >= 1


def test_scripted_goal_preempts_the_patrol(tmp_path):
    path = tmp_path / "goal.yaml"
    path.write_text(
        "name: goal\nmap: current_map\nduration: 150\nstart: {x: 28.0, y: 18.95, theta: 0.0}\nobjects: []\n"
        "waypoints: [{x: 46.0, y: 18.95, theta: 3.14159}, {x: 28.0, y: 18.95, theta: 0.0}]\n"
        "goals: [{at: 10, x: 34.0, y: 18.95, theta: 1.5708}]\n")
    sim = FastSim(str(path))
    sim.orchestrator.on_start(sim)
    reached = False
    for _ in range(int(sim.duration / sim.dt)):
        sim.step()
        r = sim.robots[99]
        if abs(r.pose.x - 34.0) < 0.3 and r.nav.mode == "IDLE":
            reached = True
    assert reached
