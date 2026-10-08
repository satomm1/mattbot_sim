"""FastSim: the whole fleet on one clock, without ROS.

Each step (dt, 10 Hz): scenario events -> orchestrator -> every robot's navigator (decisions + kinematic motion).
Every 1 / detect_hz: every robot's camera (mattbot_sim.perception.detect; other robots occlude) feeds its mapper
and its evaluator windows (no frame while the TurnGate drops it). Every 1 s: mapper blockout expiry, ledger
beliefs (shared fleet ledger), importance of new objects, scoring. Times are simulated seconds from 0.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401
from mattbot_sim.fast.checker import CheckPlanner, Evaluator, ObservationCore
from mattbot_sim.fast.fleet import FleetLedger
from mattbot_sim.fast.mapper import Mapper
from mattbot_sim.fast.navigator import FastNavigator
from mattbot_sim.fast.orchestrator import PatrolOrchestrator
from mattbot_sim.fast.params import FastParams
from mattbot_sim.fast.planning import Planner
from mattbot_sim.fast.worldview import FleetWorld
from mattbot_sim.kinematics import Pose2D, compose
from mattbot_sim.peers import peer_observations
from mattbot_sim.perception import CameraModel, DetectorNoise, TurnGate, detect
from mattbot_sim.scenario import load_scenario
from mattbot_sim.scoring import RunScorer, check_expectations, kind_name
from mattbot_sim.world import SimObject, load_maps

SINGLE_ROBOT_ID = 99  # sim.launch robot_id default
CAMERA_OFFSET = (0.127, 0.0, 0.2124)  # camera_link in base_footprint (mattbot_bringup/urdf/robot_tf_short.urdf)
EVENTS = ("STARTED", "ENDED", "ABORTED")


@dataclass
class Robot:
    id: int
    scenario: object  # this robot's view of the scenario (start, waypoints, expect)
    pose: Pose2D
    mapper: Mapper
    planner: CheckPlanner
    evaluator: Evaluator
    scorer: RunScorer
    turn_gate: TurnGate = field(default_factory=TurnGate)
    nav: FastNavigator = None
    idle_since: float = 0.0
    log: List[dict] = field(default_factory=list)
    stops: List[dict] = field(default_factory=list)


class FastSim:
    def __init__(self, scenario_path, params=None, seed=0, orchestrator=None, duration=None, map_json_dir=None,
                 roadmap_cache_dir=None, verbose=False):
        base = load_scenario(scenario_path)
        self.params = params or FastParams()
        self.params.update(base.launch_args)  # e.g. observe_detour from the scenario (explicit params win below)
        if params is not None and getattr(params, "_explicit", None):
            self.params.update(params._explicit)
        p = self.params
        self.scenario = base
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.dt = p.dt
        self.now = 0.0
        self.elapsed = 0.0
        self.duration = float(duration if duration is not None else base.duration)
        self.goals = list(base.goals)

        grid, grid_mod = load_maps(base.map, map_json_dir or imports.MAP_JSON_DIR)
        self.grid = grid
        nav_occ = np.maximum(grid.occupancy, grid_mod.occupancy)  # /navigation_map = max(map, map_mod)
        self.planner = Planner(nav_occ, (grid.origin_x, grid.origin_y), grid.resolution, 2 * p.robot_clearance,
                               p.cruising_velocity)
        self.world = FleetWorld(grid, [SimObject(o.object_id, o.class_name, o.x, o.y, o.width, o.present)
                                       for o in base.objects], now=0.0)
        self.core = ObservationCore(grid, p, cache_dir=roadmap_cache_dir)
        self.ledger = FleetLedger(p.ledger_match_radius_m, p.belief_hold_s, p.belief_decay_s)
        self.camera = CameraModel()
        self.noise = DetectorNoise(p.miss_prob, p.pos_noise_std, p.false_pos_rate)

        ids = base.robot_ids or [SINGLE_ROBOT_ID]
        self.robots: Dict[int, Robot] = {}
        for rid in ids:
            sc = load_scenario(scenario_path, rid) if base.is_fleet else base
            r = Robot(rid, sc, Pose2D(sc.start.x, sc.start.y, sc.start.theta),
                      Mapper(grid.occupancy.shape, (grid.origin_x, grid.origin_y), grid.resolution,
                             p.mapper_confirm_count, p.mapper_miss_limit, p.mapper_ttl_s, p.mapper_max_blockout_m),
                      CheckPlanner(self.core), Evaluator(p, self.camera), RunScorer())
            r.nav = FastNavigator(r, self)
            self.robots[rid] = r
        self.update_discs()

        peers = [o for o in base.objects if o.object_id in set(base.known_from_peer)]
        for d in peer_observations(peers, base.peer_id, 1, 0.0):
            self.ledger.add_peer_observation(d)

        self.events = list(base.events)
        self.orchestrator = orchestrator or PatrolOrchestrator()
        self.next_detect = 0.0
        self.next_second = 0.0
        self.wall_s = 0.0
        self.frames = None  # a list -> viz.snapshot() every frame_every s (for --gif / --png)
        self.frame_every = 2.0
        self.next_frame = 0.0

    # ---------- Services for the navigator / orchestrator ----------

    def beliefs(self):
        return self.ledger.beliefs(self.now)

    def set_goal(self, robot_id, x, y, theta):
        self.robots[robot_id].nav.set_goal(x, y, theta)

    def other_robot_positions(self, robot):
        return [(o.pose.x, o.pose.y) for o in self.robots.values() if o is not robot]

    def robot_collides(self, robot, x, y):
        """Moving robot to (x, y) would hit another robot (moves that increase the distance are allowed)."""
        limit = 2.0 * self.params.robot_radius + 0.05
        for o in self.robots.values():
            if o is robot:
                continue
            d_new = math.hypot(o.pose.x - x, o.pose.y - y)
            if d_new < limit and d_new < math.hypot(o.pose.x - robot.pose.x, o.pose.y - robot.pose.y):
                return True
        return False

    def nav_log(self, robot, text):
        robot.scorer.add_nav_log(text)
        self.note(robot, text)

    def runner_event(self, robot, text):
        self.note(robot, robot.scorer.add_runner_event(text))

    def note(self, robot, line):
        robot.log.append({"t": round(self.now, 1), "msg": line})
        if self.verbose:
            print("[%7.1f] robot %d: %s" % (self.now, robot.id, line))

    def observation_event(self, robot, name, kind, object_id, target, window_start, window_end, cost_s):
        line = robot.scorer.add_stop_event(name, kind_name(kind), object_id, self.now, (robot.pose.x, robot.pose.y),
                                           math.hypot(target[0] - robot.pose.x, target[1] - robot.pose.y))
        if line:
            self.note(robot, line)
        beliefs = self.beliefs()
        if name == "STARTED":
            robot.evaluator.start(object_id, target, self.now, beliefs)
        elif name == "ABORTED":
            robot.evaluator.abort(object_id)
        elif name == "ENDED":
            robot.planner.record_check(object_id, window_end)
            result, actions = robot.evaluator.end(object_id, self.now, beliefs)
            for action, cls, x, y, w in actions:
                if action == "confirm":
                    self.confirm(robot, cls, x, y, w)
                else:
                    self.ledger.remove_near(robot.id, self.now, cls, x, y, w)
            if result is not None:
                self.note(robot, robot.scorer.add_result(self.now, object_id, result["class_name"], result["outcome"],
                                                         result["reason"], window_start, window_end))

    def confirm(self, robot, cls, x, y, w):
        """/confirmed_objects: the robot's ledger entry, and (data_publisher) the other robots' mappers."""
        self.ledger.confirm(robot.id, self.now, cls, x, y, w)
        for o in self.robots.values():
            if o is not robot:
                o.mapper.add_from_agent(self.now, x, y, w)

    # ---------- Loop ----------

    def update_discs(self):
        self.world.robot_discs = {rid: (r.pose.x, r.pose.y, self.params.robot_radius) for rid, r in self.robots.items()}

    def apply_events(self):
        changed = False
        while self.events and self.events[0]["at"] <= self.elapsed:
            e = self.events.pop(0)
            desc = self.world.apply_event(e, self.now)
            changed = True
            for r in self.robots.values():
                self.note(r, "event: " + desc)
        if changed or not getattr(self, "_truth_set", False):
            placements = [pl.to_dict() for pl in self.world.placements]
            for r in self.robots.values():
                r.scorer.set_ground_truth(placements)
            self._truth_set = True

    def detector_frames(self):
        beliefs = self.beliefs()
        for r in self.robots.values():
            if r.turn_gate.turning(self.now):
                continue  # the detector publishes nothing while turning
            cam = compose(r.pose, Pose2D(CAMERA_OFFSET[0], CAMERA_OFFSET[1], 0.0))
            self.world.observer = r.id
            dets = detect(self.world, cam.x, cam.y, r.pose.theta, CAMERA_OFFSET[2], self.camera, self.noise, self.rng)
            self.world.observer = None
            for cls, x, y, w in r.mapper.frame(self.now, r.nav.mode, dets):
                self.confirm(r, cls, x, y, w)
            r.evaluator.frame(dets, (cam.x, cam.y, r.pose.theta, CAMERA_OFFSET[2]), beliefs)

    def every_second(self):
        beliefs = self.beliefs()
        self.core.update_objects(beliefs)
        objects = [(b.object_id, b.class_name, b.x, b.y) for b in beliefs]
        for r in self.robots.values():
            r.mapper.expire(self.now, r.nav.mode)
            if self.params.ledger_blockout:
                r.mapper.set_ledger_objects(beliefs)
            for line in r.scorer.update_ledger(self.now, objects):
                self.note(r, line)
            for line in r.scorer.update_blockouts(self.now, r.mapper.is_blocked):
                self.note(r, line)

    def step(self):
        self.apply_events()
        self.orchestrator.on_tick(self)
        for r in self.robots.values():
            mode = r.nav.mode
            r.nav.step(self.dt)
            if r.nav.mode == "IDLE" and mode != "IDLE":
                r.idle_since = self.now
            r.turn_gate.update(r.nav.om, self.now)
        self.update_discs()
        if self.now + 1e-9 >= self.next_detect:
            self.detector_frames()
            self.next_detect += 1.0 / self.params.detect_hz
        if self.now + 1e-9 >= self.next_second:
            self.every_second()
            self.next_second += 1.0
        if self.frames is not None and self.now + 1e-9 >= self.next_frame:
            from mattbot_sim.fast.viz import snapshot

            self.frames.append(snapshot(self))
            self.next_frame += self.frame_every
        self.now += self.dt
        self.elapsed = self.now

    def run(self):
        t0 = time.time()  # wall clock: run_wall_s
        self.orchestrator.on_start(self)
        n = int(round(self.duration / self.dt)) if self.duration > 0 else 0
        for _ in range(n):
            self.step()
        self.every_second()
        self.wall_s = time.time() - t0  # wall clock: run_wall_s
        return self.summaries()

    # ---------- Results ----------

    def summaries(self):
        """{robot_id: summary} with each robot's expectations (as sim_monitor writes them)."""
        out = {}
        for rid, r in self.robots.items():
            s = r.scorer.summary(self.now)
            s["scenario"] = self.scenario.name
            if self.scenario.is_fleet:
                s["robot_id"] = rid
            s["run_s"] = round(self.now, 1)
            s["run_wall_s"] = round(self.wall_s, 2)
            s["simulator"] = "fast"
            s["seed"] = self.seed
            s["navigator"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in r.nav.counters.items()}
            checks = check_expectations(s, r.scenario.expect)
            if checks:
                s["expectations"] = checks
                s["expectations_passed"] = all(c["ok"] for c in checks)
            out[rid] = s
        return out

    def results(self, robot_id):
        """The sim_monitor results JSON for one robot (summary, log, removals, observations, ledger_added, stops)."""
        r = self.robots[robot_id]
        return {"summary": self.summaries()[robot_id], "log": r.log, "removals": r.scorer.removals,
                "observations": r.scorer.results, "ledger_added": r.scorer.added, "stops": r.scorer.stop_log}

    def write_results(self, out_dir):
        import json

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for rid in self.robots:
            name = "%s%s_seed%d.json" % (self.scenario.name, "_robot%d" % rid if self.scenario.is_fleet else "", self.seed)
            path = os.path.join(out_dir, name)
            with open(path, "w") as f:
                json.dump(self.results(rid), f, indent=2)
            paths.append(path)
        return paths
