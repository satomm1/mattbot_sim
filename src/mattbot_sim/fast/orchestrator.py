"""Who sends which robot where. The simulator calls on_start(sim) once and on_tick(sim) every step.

PatrolOrchestrator does what scenario_runner.py does for each robot: send its waypoints in turn; a goal is done
when the robot has been IDLE for idle_settle_s within reach_tol_m of it (at least 3 s after sending); IDLE short
of it -> re-send (up to max_retries), then move on; pause_s (or pause_at_waypoint_s) at each waypoint; loop.
Scripted goals (scenario ``goals``) pre-empt a robot's patrol at their time; the patrol resumes afterwards.

A custom orchestrator is any class with on_start(sim) / on_tick(sim) (subclass Orchestrator); use
sim.set_goal(robot_id, x, y, theta), sim.robots, sim.now, sim.beliefs(), sim.world. For example "send the
nearest idle robot to every newly reported object". Load one with fastsim.py --orchestrator package.module:Class.
"""

import importlib
import math


class Orchestrator:
    def on_start(self, sim):
        pass

    def on_tick(self, sim):
        pass


class PatrolOrchestrator(Orchestrator):
    def __init__(self, reach_tol_m=0.5, idle_settle_s=1.5, pause_at_waypoint_s=2.0, goal_timeout_s=300.0,
                 max_retries=3):
        self.reach_tol_m = reach_tol_m
        self.idle_settle_s = idle_settle_s
        self.pause_s = pause_at_waypoint_s
        self.goal_timeout_s = goal_timeout_s
        self.max_retries = max_retries
        self.state = {}  # robot_id -> dict

    def on_start(self, sim):
        for rid, robot in sim.robots.items():
            self.state[rid] = {"idx": 0 if robot.scenario.waypoints else None, "sent_at": None, "retries": 0,
                               "wait_until": None, "scripted": None}
        self.goals = list(sim.goals)

    def _send(self, sim, rid, goal):
        st = self.state[rid]
        sim.set_goal(rid, *goal)
        st["sent_at"] = sim.now

    def on_tick(self, sim):
        while self.goals and self.goals[0]["at"] <= sim.elapsed:
            g = self.goals.pop(0)
            rids = [g["robot"]] if g["robot"] is not None else list(sim.robots)
            for rid in rids:
                if rid in self.state:
                    self.state[rid].update(scripted=(g["x"], g["y"], g["theta"]), retries=0, wait_until=None)
                    self._send(sim, rid, self.state[rid]["scripted"])
        for rid, robot in sim.robots.items():
            st = self.state[rid]
            if st["wait_until"] is not None:
                if sim.now < st["wait_until"]:
                    continue
                st["wait_until"] = None
                self._advance(robot, st)
            goal = st["scripted"] or self._waypoint(robot, st)
            if goal is None:
                continue
            if st["sent_at"] is None:
                self._send(sim, rid, goal)
                continue
            idle_for = sim.now - robot.idle_since if robot.nav.mode == "IDLE" else 0.0
            if sim.now - st["sent_at"] > 3.0 and idle_for > self.idle_settle_s:
                dist = math.hypot(robot.pose.x - goal[0], robot.pose.y - goal[1])
                if dist <= self.reach_tol_m or st["retries"] >= self.max_retries:
                    if dist > self.reach_tol_m:
                        sim.runner_event(robot, "give_up goal %.1f m short" % dist)
                    pause = self.pause_s
                    if st["scripted"] is None:
                        pauses = robot.scenario.waypoint_pauses
                        if st["idx"] < len(pauses) and pauses[st["idx"]] is not None:
                            pause = pauses[st["idx"]]
                    st["wait_until"] = sim.now + pause
                    st["sent_at"] = None
                else:
                    st["retries"] += 1
                    sim.runner_event(robot, "resend goal %.1f m short" % dist)
                    st["sent_at"] = None
            elif sim.now - st["sent_at"] > self.goal_timeout_s:
                sim.runner_event(robot, "resend goal timed out")
                st["sent_at"] = None

    def _waypoint(self, robot, st):
        if st["idx"] is None:
            return None
        w = robot.scenario.waypoints[st["idx"]]
        return (w.x, w.y, w.theta)

    def _advance(self, robot, st):
        st["retries"] = 0
        if st["scripted"] is not None:
            st["scripted"] = None  # back to the patrol, at the waypoint it was heading for
            return
        if st["idx"] is None:
            return
        st["idx"] += 1
        if st["idx"] >= len(robot.scenario.waypoints):
            st["idx"] = 0 if robot.scenario.loop else None


def load_orchestrator(spec):
    """'package.module:ClassName' -> instance."""
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError("orchestrator must be package.module:ClassName, got %r" % spec)
    return getattr(importlib.import_module(module), name)()
