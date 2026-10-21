"""Task orchestrator for experiments: a fixed list of trips, released over time, given to robots.

Released tasks wait in a FIFO; the first one goes to the nearest available robot (straight-line distance). A
robot is available when it has no task and is not dwelling at its last destination; a robot out on a scout
counts as available (the task pre-empts the scout: FastNavigator.set_goal cancels it). A task is done when the
robot is IDLE for idle_settle_s, at least 3 s after the goal was sent, within reach_tol_m of the destination
(the PatrolOrchestrator rule); a robot that went IDLE short of it gets the goal again, up to max_retries, and a
task that takes longer than task_timeout_s or runs out of retries fails. The run ends (sim.done) when every task
is done or failed, or, with end_at_horizon, only at the simulator's duration (tasks still open are unfinished).

A robot that runs out of retries because it cannot plan (its plan failures grew during the task, e.g. ledger
blockouts closed it into a pocket after a detour) is stuck, not the task: the task goes back to the front of the
queue for another robot (up to max_handoffs times, then it fails) and the stuck robot gets no task for
stuck_backoff_s.
"""

import math

from mattbot_sim.fast.orchestrator import Orchestrator


class TaskOrchestrator(Orchestrator):
    def __init__(self, tasks, dwell_s=5.0, reach_tol_m=0.5, idle_settle_s=1.5, max_retries=3, task_timeout_s=900.0,
                 end_at_horizon=False, stuck_backoff_s=120.0, max_handoffs=3):
        self.end_at_horizon = bool(end_at_horizon)
        self.stuck_backoff_s = float(stuck_backoff_s)
        self.max_handoffs = int(max_handoffs)
        self.stuck_events = 0  # times a robot was found stuck (its task handed to another robot)
        self.tasks = [dict(t, status="waiting", robot=None, assigned_s=None, done_s=None, retries=0, handoffs=0)
                      for t in tasks]
        self.dwell_s = float(dwell_s)
        self.reach_tol_m = reach_tol_m
        self.idle_settle_s = idle_settle_s
        self.max_retries = max_retries
        self.task_timeout_s = task_timeout_s

    def on_start(self, sim):
        self.pending = []  # released, not yet assigned (FIFO)
        self.next_release = 0
        self.state = {rid: {"task": None, "sent_at": None, "free_at": 0.0, "stuck_until": 0.0, "failures_at": 0}
                      for rid in sim.robots}

    # ---------- Helpers ----------

    def _send(self, sim, rid, task):
        st = self.state[rid]
        st["sent_at"] = sim.now
        sim.set_goal(rid, task["x"], task["y"], task["theta"])

    def _finish(self, sim, rid, task, status):
        task["status"] = status
        task["done_s"] = round(sim.now, 2)
        st = self.state[rid]
        st["task"] = None
        st["free_at"] = sim.now + (self.dwell_s if status == "done" else 0.0)
        r = sim.robots[rid]
        sim.runner_event(r, "task %d %s at %s" % (task["id"], status, task["landmark"]))

    def available(self, sim, rid):
        st = self.state[rid]
        return st["task"] is None and sim.now >= st["free_at"] and sim.now >= st["stuck_until"]

    def _hand_off(self, sim, rid, task):
        """The robot cannot plan: put its task back at the front of the queue and rest the robot."""
        st = self.state[rid]
        st["task"] = None
        st["stuck_until"] = sim.now + self.stuck_backoff_s
        self.stuck_events += 1
        task.update(status="released", robot=None, assigned_s=None, retries=0, handoffs=task["handoffs"] + 1)
        self.pending.insert(0, task)
        sim.runner_event(sim.robots[rid], "robot cannot plan: task %d handed to another robot" % task["id"])

    def may_scout(self, sim, robot):
        """FastSim.scout_tick asks before executing a scout: only with no task waiting and none of its own."""
        return not self.pending and self.state[robot.id]["task"] is None

    # ---------- Tick ----------

    def on_tick(self, sim):
        while self.next_release < len(self.tasks) and self.tasks[self.next_release]["release_s"] <= sim.elapsed:
            task = self.tasks[self.next_release]
            task["status"] = "released"
            self.pending.append(task)
            self.next_release += 1

        for rid, st in self.state.items():
            task = st["task"]
            if task is None:
                continue
            r = sim.robots[rid]
            if sim.now - task["assigned_s"] > self.task_timeout_s:
                r.nav.stop()
                self._finish(sim, rid, task, "failed")
                continue
            settled = (r.nav.mode == "IDLE" and sim.now - r.idle_since >= self.idle_settle_s
                       and sim.now - st["sent_at"] >= 3.0)
            if not settled:
                continue
            if math.hypot(r.pose.x - task["x"], r.pose.y - task["y"]) <= self.reach_tol_m:
                self._finish(sim, rid, task, "done")
            elif task["retries"] < self.max_retries:
                task["retries"] += 1
                sim.runner_event(r, "resend goal %d %.1f m short" % (task["id"], math.hypot(
                    r.pose.x - task["x"], r.pose.y - task["y"])))
                self._send(sim, rid, task)
            elif (r.nav.counters.get("plan_failures", 0) > st["failures_at"]
                  and task["handoffs"] < self.max_handoffs):
                self._hand_off(sim, rid, task)
            else:
                self._finish(sim, rid, task, "failed")

        while self.pending:
            free = [rid for rid in self.state if self.available(sim, rid)]
            if not free:
                break
            task = self.pending.pop(0)
            rid = min(free, key=lambda i: (math.hypot(sim.robots[i].pose.x - task["x"],
                                                      sim.robots[i].pose.y - task["y"]), i))
            task.update(status="assigned", robot=rid, assigned_s=round(sim.now, 2))
            self.state[rid]["task"] = task
            self.state[rid]["failures_at"] = sim.robots[rid].nav.counters.get("plan_failures", 0)
            self._send(sim, rid, task)

        if not self.end_at_horizon and self.next_release >= len(self.tasks) and not self.pending and all(
                st["task"] is None for st in self.state.values()):
            sim.done = True


def orchestrator_for(spec, instance):
    """The TaskOrchestrator of an experiment spec for one instance."""
    return TaskOrchestrator(instance["tasks"], dwell_s=spec.tasks["dwell_s"], end_at_horizon=spec.end_at_horizon)
