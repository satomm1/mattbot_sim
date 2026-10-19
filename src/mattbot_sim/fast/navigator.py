"""The navigator state machine of mattbot_navigation/scripts/localize_and_navigate.py, without ROS.

Modes and rules follow the navigator (references are to that file); motion is kinematic:
  - ALIGN, OBSERVE_TURN, PARK_HEADING: the navigator's HeadingController, integrated at dt (w clamped as sim_world)
  - PARK_POSE: its PoseController with unicycle integration
  - TRACK: the robot is exactly on the smoothed trajectory at the TrajectoryTracker's reference time (soft start:
    the reference trails the plan clock by soft_start_s / 2), i.e. perfect tracking with the tracker's timing.
Observation flow (request after each plan, trigger, turn / dwell, resume), detours (start, arrival, deadline,
abandon) and the TRACK check order (near goal, observation due, detour deadline, path validity, out of time) are
the navigator's, so its quirks (e.g. a detour whose leave point is at the goal is dropped) carry over.
Not modelled: waypoint deadlines and the localization recovery chain, stall detection, people, localization error.
"""

import math

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401
from mattbot_sim.kinematics import Pose2D, step, wrap_angle
from navigation_utils import plan_start_heading
from dds_utils.intents import Intent
from navigation_utils.observation_policy import AT_GOAL, AT_START, DETOUR, OPPORTUNISTIC, REPLAN, check_etas, yields_to
from navigation_utils.trackers import HeadingController, PoseController

OBSERVE_MODES = ("OBSERVE_TURN", "OBSERVE_DWELL")
ENDPOINT_KINDS = (AT_START, AT_GOAL)
NAV = "[Navigator] "


def reference_time(t, soft_start_s):
    """TrajectoryTracker._reference_time: smoothstep ramp, then lagging the plan clock by soft_start_s / 2."""
    T = soft_start_s
    t = max(0.0, float(t))
    if T <= 0.0:
        return t
    if t <= T:
        u = t / T
        return T * (u ** 3 - 0.5 * u ** 4)
    return t - 0.5 * T


class FastNavigator:
    def __init__(self, robot, sim):
        self.robot = robot  # .id, .pose (Pose2D), .planner (CheckPlanner), .mapper
        self.sim = sim  # .now, .params, .planner (Planner), .beliefs(), .other_robots(id), event / log hooks
        p = sim.params
        self.p = p
        self.robot_d = 2.0 * p.robot_clearance
        self.heading = HeadingController(om_max=3)
        self.park = PoseController(1.0, 2.0, 1.0, p.v_max, 3)
        self.mode = "IDLE"
        self.goal = None  # (x, y, theta)
        self.plan = None  # times, traj of the current trajectory
        self.t_plan = 0.0  # plan clock (s since TRACK started)
        self.duration = 0.0
        self.park_goal = None
        self.th_init = 0.0
        self.aligned_since = None
        self.stops = []  # pending observation stops of this trajectory
        self.observe_goal = None
        self.observed_this_goal = set()
        self.detour_skipped = set()
        self.start_stop = None  # AT_START / AT_GOAL entries of the current plan
        self.goal_stop = None
        self.yielded = {}  # object_id -> robot whose earlier check a detour of ours gave way to (this goal)
        self.progress = 0.0  # plan time reached on the current trajectory (frozen outside TRACK)
        self.intents_dirty = True
        self.intents_sent_at = -1e9
        self.detour = None
        self.observe_stop = None
        self.observe_queue = []
        self.observe_target = None
        self.phase_start = 0.0
        self.window_start = 0.0
        self.resuming = False
        self.prev_om = 0.0
        self.om = 0.0  # last commanded angular velocity (TurnGate)
        self.blocked_s = 0.0
        self.valid_version = None
        self.counters = {"replans": 0, "plan_failures": 0, "robot_blocked_s": 0.0, "robot_replans": 0}

    # ---------- Helpers ----------

    @property
    def pose(self):
        return self.robot.pose

    def log(self, text, *args):
        self.sim.nav_log(self.robot, NAV + (text % args if args else text))

    def switch_mode(self, new_mode):
        if self.mode in OBSERVE_MODES and new_mode not in OBSERVE_MODES and not self.resuming:
            self.abort_observation()
        if self.mode == "ALIGN" and new_mode != "ALIGN":
            self.aligned_since = None
        if new_mode == "TRACK":
            self.t_plan = 0.0
            self.blocked_s = 0.0
        self.mode = new_mode
        self.intents_dirty = True

    def active_goal(self):
        if self.detour is not None:
            if self.goal != self.detour["goal"]:
                self.cancel_detour("goal changed")
            else:
                s = self.detour["stop"]
                return s.x, s.y, self.detour["heading"]
        return self.goal

    def near_goal(self):
        g = self.active_goal()
        return g is not None and math.hypot(self.pose.x - g[0], self.pose.y - g[1]) < self.p.near_thresh

    def aligned(self):
        return abs(wrap_angle(self.pose.theta - self.th_init)) < self.p.theta_start_thresh

    # ---------- Goals ----------

    def set_goal(self, x, y, theta):
        """/external_goal (goal_callback)."""
        goal = (float(x), float(y), float(theta))
        if goal == self.goal:
            if self.mode == "IDLE":
                self.replan()
            return
        self.switch_mode("IDLE")
        if not self.sim.planner.grid().is_free((x, y)):
            self.log("invalid goal (%.2f, %.2f)", x, y)
            return
        self.goal = goal
        self.replan()

    def stop(self):
        """/stop."""
        self.cancel_detour("stop")
        self.start_stop = self.goal_stop = None
        self.switch_mode("IDLE")
        self.goal = None

    def replan(self, robots=()):
        """localize_and_navigate.replan(); robots: other robots to plan around (blocked by a robot)."""
        if self.mode == "TRACK":
            return
        g = self.active_goal()
        if g is None:
            return
        self.counters["replans"] += 1
        self.start_stop = self.goal_stop = None
        planner = self.sim.planner
        grid = planner.grid(self.robot.mapper.object_layer(), (self.robot.id, self.robot.mapper.version))
        start = planner.start_cell(grid, self.pose.x, self.pose.y)
        if start is None:
            if self.detour is not None:
                self.abandon_detour("no free start cell")
            else:
                self.switch_mode("IDLE")
            return
        plan = planner.plan(start, (g[0], g[1]), grid, robots=robots, goal_snap_m=0.4 if self.detour else 0.0)
        if plan is None:
            self.counters["plan_failures"] += 1
            if self.detour is not None:
                self.abandon_detour("no path to the viewpoint")
                return
            self.log("Planning failed; clearing the goal")
            self.goal = None
            self.switch_mode("IDLE")
            return
        if plan.short:
            if self.detour is not None:
                self.arrive_at_detour_viewpoint()
            else:
                # localize_and_navigate.replan: park at the end of this short path with the goal heading
                end = plan.path[-1] if plan.path else (g[0], g[1])
                self.park_goal = (float(end[0]), float(end[1]), float(g[2]))
                self.log("Path too short to track; parking at (%.2f, %.2f)", end[0], end[1])
                self.enter_park_pose()
            return
        self.plan = (plan.times, plan.traj)
        self.progress = 0.0
        self.duration = float(plan.times[-1])
        self.park_goal = (float(plan.traj[-1, 0]), float(plan.traj[-1, 1]), float(plan.traj[-1, 2]))
        self.th_init = plan.th_init
        self.heading.load_goal(self.th_init)
        self.valid_version = self.robot.mapper.version
        self.request_observation_stops()
        if self.start_stop is not None:  # look from here before leaving, then ALIGN (load_remaining_plan(0))
            entry, self.start_stop = self.start_stop, None
            self.aligned_since = None
            if self.start_observation_stop(entry):
                return
        self.switch_mode("ALIGN")
        if self.aligned():
            self.aligned_since = self.sim.now

    def enter_park_pose(self):
        g = self.goal
        if self.park_goal is None and g is not None:
            self.park_goal = g
        if self.park_goal is None:
            self.switch_mode("IDLE")
            return
        self.park.load_goal(*self.park_goal)
        self.switch_mode("PARK_POSE")

    # ---------- Observation stops ----------

    def request_observation_stops(self):
        self.stops = []
        if not self.p.observe or self.detour is not None:
            return
        first_plan = self.goal != self.observe_goal
        if first_plan:
            self.observe_goal = self.goal
            self.observed_this_goal = set()
            self.detour_skipped = set()
            self.yielded = {}
        times, traj = self.plan
        stride = self.p.observe_path_stride
        ks = list(range(0, len(traj), stride))
        if ks[-1] != len(traj) - 1:
            ks.append(len(traj) - 1)  # end at the park pose
        path_xy = traj[ks, :2]
        req_times = [float(times[k]) for k in ks]
        start_pose = (self.pose.x, self.pose.y, self.pose.theta) if first_plan else None
        stops = self.robot.planner.select(path_xy, self.sim.beliefs(), self.sim.now,
                                          exclude=self.observed_this_goal | self.detour_skipped | self.live_yielded(),
                                          start_pose=start_pose, goal_theta=self.goal[2], times=req_times,
                                          peer_intents=self.sim.peer_intents(self.robot), robot_id=self.robot.id)
        have_detour = False
        endpoints = []
        for s in stops:
            if s.path_index < 0:
                continue
            idx = min(s.path_index * stride, len(traj) - 1)
            entry = {"idx": idx, "t": float(times[idx]), "stop": s, "detour": False}
            if s.kind in ENDPOINT_KINDS:
                entry["goal"] = self.goal
                if s.kind == AT_START:
                    entry["idx"], entry["t"] = 0, 0.0
                    self.start_stop = entry
                else:
                    entry["idx"], entry["t"] = len(traj) - 1, float(times[-1])
                    self.goal_stop = entry
                endpoints.append(entry)
                continue
            if s.kind == DETOUR:
                if not self.p.observe_detour or have_detour:
                    continue
                have_detour = True
                entry["detour"] = True
                entry["leave"] = (float(traj[idx, 0]), float(traj[idx, 1]), float(traj[idx, 2]))
            elif s.kind != OPPORTUNISTIC:
                continue
            self.stops.append(entry)
        self.stops.sort(key=lambda e: e["idx"])
        planned = sorted(self.stops + endpoints, key=lambda e: e["idx"])
        for entry, eta in zip(planned, check_etas([e["stop"] for e in planned], req_times)):
            entry["eta"] = self.sim.now + eta
        self.intents_dirty = True
        if planned:
            prefix = {DETOUR: "detour:", AT_START: "at_start:", AT_GOAL: "at_goal:"}
            self.log("Observation stops planned: %s", ", ".join(
                prefix.get(e["stop"].kind, "") + "+".join(e["stop"].object_ids) for e in planned))

    def observation_due(self):
        if not self.stops:
            return False
        entry = self.stops[0]
        s = entry["stop"]
        sx, sy, sh = entry["leave"] if entry["detour"] else (s.x, s.y, s.heading)
        along = (self.pose.x - sx) * math.cos(sh) + (self.pose.y - sy) * math.sin(sh)
        if math.hypot(self.pose.x - sx, self.pose.y - sy) < self.p.observe_trigger_m and along >= -0.05:
            return True
        if self.t_plan > entry["t"] + self.p.stop_passed_slack_s:
            self.log("Passed observation stop for %s without reaching it; skipping", list(s.object_ids))
            self.stops.pop(0)
        return False

    def start_observation_stop(self, entry=None):
        """False if every target was already checked for this goal (nothing started)."""
        entry = entry if entry is not None else self.stops.pop(0)
        s = entry["stop"]
        queue = [(oid, t) for oid, t in zip(s.object_ids, s.targets) if oid not in self.observed_this_goal]
        if queue:
            due = self.revalidate(entry, [q[0] for q in queue], self.p.observe_dwell_s * len(queue))
            queue = [q for q in queue if q[0] in due]
        if not queue:
            return False
        self.observe_stop = entry
        self.observe_queue = queue
        self.next_observe_target()
        return True

    def start_goal_check(self):
        """At the park pose: look at this plan's AT_GOAL objects before the final heading turn."""
        entry, self.goal_stop = self.goal_stop, None
        if entry is None or entry["goal"] != self.goal:
            return False
        return self.start_observation_stop(entry)

    def finish_park_pose(self):
        self.heading.load_goal(self.goal[2] if self.goal else self.park_goal[2])
        self.phase_start = self.sim.now
        self.mode = "PARK_HEADING"

    def next_observe_target(self):
        self.observe_target = self.observe_queue.pop(0)
        _oid, (tx, ty) = self.observe_target
        self.heading.load_goal(math.atan2(ty - self.pose.y, tx - self.pose.x))
        self.phase_start = self.sim.now
        self.switch_mode("OBSERVE_TURN")

    def event(self, name, window_end=0.0):
        oid, target = self.observe_target
        kind = self.observe_stop["stop"].kind if self.observe_stop else OPPORTUNISTIC
        self.sim.observation_event(self.robot, name, kind, oid, target, self.window_start, window_end,
                                   self.observe_stop["stop"].cost_s if self.observe_stop else 0.0)

    def abort_observation(self):
        if self.observe_target is not None:
            self.event("ABORTED", window_end=self.sim.now)
        self.observe_stop = None
        self.observe_queue = []
        self.observe_target = None

    def observe_step(self):
        now = self.sim.now
        if self.mode == "OBSERVE_TURN":
            err = wrap_angle(self.heading.th_g - self.pose.theta)
            if abs(err) < self.p.theta_start_thresh:
                self.window_start = now
                self.event("STARTED")
                self.phase_start = now
                self.mode = "OBSERVE_DWELL"
            elif now - self.phase_start > self.p.observe_turn_timeout_s:
                self.event("ABORTED", window_end=now)
                self.finish_observation_target(done=False)
        elif self.mode == "OBSERVE_DWELL" and now - self.phase_start >= self.p.observe_dwell_s:
            self.event("ENDED", window_end=now)
            self.finish_observation_target(done=True)

    def finish_observation_target(self, done):
        oid = self.observe_target[0]
        if done:
            self.observed_this_goal.add(oid)
        elif self.observe_stop and (self.observe_stop["detour"] or self.observe_stop["stop"].kind in ENDPOINT_KINDS):
            self.detour_skipped.add(oid)
        if self.observe_queue:
            self.next_observe_target()
            return
        entry, self.observe_stop, self.observe_target = self.observe_stop, None, None
        self.resuming = True
        try:
            if entry["stop"].kind == AT_START:
                self.load_remaining_plan(0)
            elif entry["stop"].kind == AT_GOAL:
                self.finish_park_pose()
            elif entry["stop"].resume == REPLAN:
                self.detour = None
                self.log("Detour done; replanning to the goal")
                self.switch_mode("IDLE")
                self.replan()
            else:
                self.load_remaining_plan(self.nearest_plan_index(entry["idx"]))
        finally:
            self.resuming = False

    def nearest_plan_index(self, idx, window=40):
        traj = self.plan[1]
        lo, hi = max(idx - window, 0), min(idx + window + 1, len(traj))
        d = np.hypot(traj[lo:hi, 0] - self.pose.x, traj[lo:hi, 1] - self.pose.y)
        return lo + int(np.argmin(d))

    def load_remaining_plan(self, idx):
        times, traj = self.plan
        rest, t_new = traj[idx:], times[idx:] - times[idx]
        if len(rest) < 4:
            self.enter_park_pose()
            return
        self.plan = (t_new, rest)
        self.progress = 0.0
        self.duration = float(t_new[-1])
        for e in self.stops:
            e["idx"] -= idx
            e["t"] -= float(times[idx])
        self.th_init = float(plan_start_heading(None, rest, v_min=0.05))
        self.heading.load_goal(self.th_init)
        self.switch_mode("ALIGN")
        self.aligned_since = self.sim.now if self.aligned() else None

    # ---------- Detours ----------

    def revalidate(self, entry, object_ids, cost_s):
        """localize_and_navigate._revalidate: ids still due a check now."""
        s = entry["stop"]
        results = self.robot.planner.revalidate(s.kind, object_ids, self.sim.beliefs(), self.sim.now,
                                                entry.get("eta"), cost_s)
        due = set()
        for oid, (ok, reason) in zip(object_ids, results):
            if ok:
                due.add(oid)
            else:
                self.log("Skipping check of %s (%s)", oid, reason)
        return due

    def start_detour(self):
        entry = self.stops[0]
        winners = self.detour_winners(entry["stop"], self.entry_eta(entry))
        if winners:
            self.stops.pop(0)
            self.record_yield(entry["stop"], winners)
            return
        if not self.revalidate(entry, list(entry["stop"].object_ids), entry["stop"].cost_s):
            self.stops.pop(0)
            self.log("Detour to check %s dropped: no longer due", "+".join(entry["stop"].object_ids))
            return
        entry = self.stops.pop(0)
        displaced = self.stops + ([self.goal_stop] if self.goal_stop is not None else [])
        self.stops = []
        s = entry["stop"]
        tx, ty = s.targets[0]
        self.detour = {"entry": entry, "stop": s, "goal": self.goal, "heading": math.atan2(ty - s.y, tx - s.x),
                       "deadline": self.sim.now + self.p.observe_detour_timeout_factor * float(s.cost_s) + 20.0,
                       "displaced": displaced}
        self.log("Detour to (%.2f, %.2f) to check %s (est. %.0f s)", s.x, s.y, "+".join(s.object_ids), s.cost_s)
        self.switch_mode("IDLE")
        self.replan()

    def arrive_at_detour_viewpoint(self):
        self.log("Reached detour viewpoint")
        if not self.start_observation_stop(self.detour["entry"]):
            self.log("Detour done; replanning to the goal (targets already checked)")
            self.detour = None
            self.switch_mode("IDLE")
            self.replan()

    def abandon_detour(self, reason):
        if self.detour is None:
            return
        ids = list(self.detour["stop"].object_ids)
        self.log("Detour to check %s abandoned (%s); replanning to the goal", "+".join(ids), reason)
        self.detour = None
        self.detour_skipped.update(ids)
        self.switch_mode("IDLE")
        self.replan()

    def cancel_detour(self, reason):
        if self.detour is not None:
            self.log("Detour cancelled (%s)", reason)
            self.detour = None

    # ---------- Path validity ----------

    def path_still_valid(self):
        """Re-checked when the robot's object layer changes (the result cannot change otherwise)."""
        mapper = self.robot.mapper
        if self.valid_version == mapper.version:
            return True
        self.valid_version = mapper.version
        grid = self.sim.planner.object_grid(mapper.object_layer(), max(self.robot_d - 2.0 * self.p.path_check_tolerance_m, 0))
        traj = self.plan[1]
        for x, y in traj[:, :2]:
            if math.hypot(x - self.pose.x, y - self.pose.y) < self.robot_d:
                continue
            if not grid.is_free((x, y)):
                self.log("Path no longer valid...")
                return False
        return True

    # ---------- Main step ----------

    def step(self, dt):
        """One 10 Hz navigator cycle: decisions, then motion."""
        now = self.sim.now
        if self.mode == "ALIGN":
            if self.aligned_since is not None:
                if now - self.aligned_since >= self.p.post_align_pause_s:
                    self.aligned_since = None
                    self.switch_mode("TRACK")
            elif self.aligned():
                self.aligned_since = now
        elif self.mode == "TRACK":
            self.progress = self.t_plan
            if not self.check_detour_yields():
                self.track_checks()
        elif self.mode in OBSERVE_MODES:
            self.observe_step()
        elif self.mode == "PARK_POSE":
            gx, gy, _ = self.park_goal
            if math.hypot(self.pose.x - gx, self.pose.y - gy) < self.p.park_pose_thresh:
                if not self.start_goal_check():
                    self.finish_park_pose()
        elif self.mode == "PARK_HEADING":
            if abs(wrap_angle(self.heading.th_g - self.pose.theta)) < self.p.theta_goal_thresh:
                self.goal = None
                self.park_goal = None
                self.switch_mode("IDLE")
        self.move(dt)
        self.publish_intents()

    # ---------- Fleet coordination (localize_and_navigate "Fleet coordination") ----------

    def peer_intents(self, object_id):
        return [i for i in self.sim.peer_intents(self.robot) if i.object_id == object_id]

    def live_yielded(self):
        self.yielded = {oid: rid for oid, rid in self.yielded.items()
                        if any(i.robot_id == rid for i in self.peer_intents(oid))}
        return set(self.yielded)

    def overhead_s(self):
        if self.mode == "TRACK":
            return 0.0
        extra = self.p.post_align_pause_s
        if self.mode in OBSERVE_MODES:
            extra += self.p.observe_dwell_s * (len(self.observe_queue) + 1)
        return extra

    def pending_etas(self):
        if self.mode in ("IDLE", "PARK_HEADING"):
            return []
        base = self.sim.now + self.overhead_s()
        out, delay = [], 0.0
        on_path = [] if self.mode == "PARK_POSE" else list(self.stops)
        for e in on_path + ([self.goal_stop] if self.goal_stop is not None else []):
            t = self.duration if e is self.goal_stop else e["t"]
            eta = base + max(t - self.progress, 0.0) + delay
            if e.get("detour"):
                eta += float(e["stop"].to_view_s)
            out.append((e, eta))
            delay += float(e["stop"].cost_s)
        return out

    def entry_eta(self, entry):
        for e, eta in self.pending_etas():
            if e is entry:
                return eta
        return float(entry.get("eta", self.sim.now))

    def planned_checks(self):
        now = self.sim.now
        out = []
        if self.mode in OBSERVE_MODES and self.observe_stop is not None:
            ids = ([self.observe_target[0]] if self.observe_target else []) + [q[0] for q in self.observe_queue]
            out += [(self.observe_stop["stop"].kind, oid, now, True) for oid in ids]
        if self.start_stop is not None:
            out += [(AT_START, oid, now, False) for oid in self.start_stop["stop"].object_ids]
        if self.detour is not None:
            if self.mode not in OBSERVE_MODES:
                eta = now + self.overhead_s() + max(self.duration - self.progress, 0.0)
                out += [(DETOUR, oid, eta, True) for oid in self.detour["stop"].object_ids]
            later = float(self.detour["stop"].cost_s)
            for e in self.detour["displaced"]:
                out += [(e["stop"].kind, oid, float(e.get("eta", now)) + later, False) for oid in e["stop"].object_ids]
        else:
            for e, eta in self.pending_etas():
                if e.get("yielded") is None:
                    out += [(e["stop"].kind, oid, eta, False) for oid in e["stop"].object_ids]
        return out

    def publish_intents(self):
        if not self.p.observe_coordinate or not self.p.observe:
            return
        now = self.sim.now
        if not self.intents_dirty and now - self.intents_sent_at < 1.0:
            return
        seen, intents = set(), []
        for kind, oid, eta, committed in self.planned_checks():
            if oid in seen:
                continue
            seen.add(oid)
            intents.append(Intent(oid, self.robot.id, kind, eta, eta + max(30.0, 0.5 * (eta - now)), committed))
        self.sim.intents.announce(self.robot.id, intents, now)
        self.intents_dirty = False
        self.intents_sent_at = now

    def detour_winners(self, stop, eta, committed=False, remaining_s=None):
        if not self.p.observe_coordinate:
            return []
        winners = []
        for oid in stop.object_ids:
            w = yields_to(DETOUR, eta, self.robot.id, self.peer_intents(oid), self.p.observe_intent_margin_s,
                          self.p.observe_intent_commit_s, committed, remaining_s)
            if w is None:
                return []
            winners.append((oid, w))
        return winners

    def record_yield(self, stop, winners):
        for oid, w in winners:
            self.yielded[oid] = w.robot_id
        self.log("Detour to check %s yielded to robot %s", "+".join(stop.object_ids),
                 ",".join(sorted({str(w.robot_id) for _oid, w in winners})))
        self.intents_dirty = True

    def check_detour_yields(self):
        if not self.p.observe_coordinate:
            return False
        if self.detour is not None:
            remaining = max(self.duration - self.progress, 0.0)
            winners = self.detour_winners(self.detour["stop"], self.sim.now + remaining, committed=True,
                                          remaining_s=remaining)
            if winners:
                self.record_yield(self.detour["stop"], winners)
                self.detour = None
                self.switch_mode("IDLE")
                self.replan()
                return True
            return False
        for e, eta in self.pending_etas():
            if e.get("detour"):
                winners = self.detour_winners(e["stop"], eta)
                yielded = winners[0][1].robot_id if winners else None
                if yielded != e.get("yielded"):
                    e["yielded"] = yielded
                    self.intents_dirty = True
        return False

    def track_checks(self):
        if self.near_goal():
            if self.detour is not None:
                self.arrive_at_detour_viewpoint()
            else:
                self.enter_park_pose()
            return
        if self.observation_due():
            if self.stops[0]["detour"]:
                self.start_detour()
            else:
                self.start_observation_stop()
            return
        if self.detour is not None and self.sim.now > self.detour["deadline"]:
            self.abandon_detour("took too long")
            return
        if not self.path_still_valid():
            self.switch_mode("IDLE")
            self.replan()
            return
        if self.t_plan > 1.2 * self.duration:
            g = self.active_goal()
            if g is not None and math.hypot(self.pose.x - g[0], self.pose.y - g[1]) > 0.5:
                self.log("replanning because out of time")
                self.switch_mode("IDLE")
                self.replan()
            elif self.detour is not None:
                self.arrive_at_detour_viewpoint()
            else:
                self.enter_park_pose()

    # ---------- Motion ----------

    def turn(self, dt):
        _v, om = self.heading.compute_control(self.pose.theta, self.sim.now, prev_om=self.prev_om)
        return float(np.clip(om, -self.p.w_max, self.p.w_max))

    def move(self, dt):
        r = self.robot
        om = 0.0
        if self.mode == "ALIGN" and self.aligned_since is None:
            om = self.turn(dt)
            r.pose = step(r.pose, 0.0, om, dt)
        elif self.mode in ("OBSERVE_TURN", "PARK_HEADING"):
            om = self.turn(dt)
            r.pose = step(r.pose, 0.0, om, dt)
        elif self.mode == "PARK_POSE":
            v, om = self.park.compute_control(r.pose.x, r.pose.y, r.pose.theta, self.sim.now)
            v = float(np.clip(v, -self.p.v_max, self.p.v_max))
            om = float(np.clip(om, -self.p.w_max, self.p.w_max))
            nxt = step(r.pose, v, om, dt)
            if not self.sim.robot_collides(r, nxt.x, nxt.y):
                r.pose = nxt
        elif self.mode == "TRACK":
            times, traj = self.plan
            t_next = self.t_plan + dt
            tr = min(reference_time(t_next, self.p.soft_start_s), float(times[-1]))
            x = float(np.interp(tr, times, traj[:, 0]))
            y = float(np.interp(tr, times, traj[:, 1]))
            th = float(np.interp(tr, times, np.unwrap(traj[:, 2])))
            if self.sim.robot_collides(r, x, y):
                self.blocked_s += dt
                self.counters["robot_blocked_s"] += dt
                if self.blocked_s >= self.p.block_wait_s:
                    self.blocked_s = 0.0
                    self.counters["robot_replans"] += 1
                    self.log("blocked by another robot; replanning around it")
                    self.switch_mode("IDLE")
                    self.replan(robots=self.sim.other_robot_positions(r))
            else:
                self.blocked_s = 0.0
                self.t_plan = t_next
                om = wrap_angle(th - r.pose.theta) / dt
                r.pose = Pose2D(x, y, wrap_angle(th))
        self.prev_om = om
        self.om = om
