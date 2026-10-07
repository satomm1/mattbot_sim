#!/usr/bin/env python3
"""Scores a sim run: compares the ledger and the observation outcomes with the ground truth.

Inputs: /sim/ground_truth (sim_world), /object_beliefs (active ledger objects), /observation/events,
        /observation/results, /object_map (occupancy_grid_mapper's object blockouts),
        /sim/runner_event (scenario_runner goal re-sends), /rosout_agg (navigator detour log lines:
        abandons and cancels publish no ObservationEvent)
Logs each scored event and, on shutdown, prints a summary (with the scenario's ``expect:`` checks) and
writes it to ~results_dir/<scenario>_<time>.json (summary + event log).
"""

import json
import os
import time

import numpy as np
import rospy
from mattbot_dds.msg import ObjectBeliefArray, ObservationEvent, ObservationResult
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from mattbot_sim.clock import wait_for_clock
from mattbot_sim.scenario import load_scenario
from mattbot_sim.scoring import OUTCOMES, RateTracker, RunScorer, check_expectations, kind_name

EVENT_NAMES = {ObservationEvent.STARTED: "STARTED", ObservationEvent.ENDED: "ENDED", ObservationEvent.ABORTED: "ABORTED"}


class SimMonitor:
    def __init__(self):
        rospy.init_node("sim_monitor")
        self.scenario = load_scenario(rospy.get_param("/sim/scenario_file"))
        self.results_dir = rospy.get_param("~results_dir", "")
        self.scorer = RunScorer(match_radius=float(rospy.get_param("~match_radius_m", 1.0)))
        self.log = []
        self.t_start = wait_for_clock()  # scoring uses ROS time (simulated with /use_sim_time)
        self.wall_start = time.time()  # wall clock: run_wall_s, to see the speedup
        self.is_blocked = None  # lookup into the latest /object_map

        rospy.Subscriber("/sim/ground_truth", String, self.truth_callback, queue_size=10)
        rospy.Subscriber("/object_beliefs", ObjectBeliefArray, self.beliefs_callback, queue_size=10)
        rospy.Subscriber("/observation/events", ObservationEvent, self.stop_callback, queue_size=50)
        rospy.Subscriber("/observation/results", ObservationResult, self.result_callback, queue_size=50)
        rospy.Subscriber("/object_map", OccupancyGrid, self.object_map_callback, queue_size=1)
        rospy.Subscriber("/sim/runner_event", String, self.runner_callback, queue_size=10)
        self.navigator_name = rospy.get_param("~navigator_node", "/navigator_node")
        rospy.Subscriber("/rosout_agg", Log, self.rosout_callback, queue_size=200)
        # Loop rates in ROS time: the navigator publishes nav_vel every 10 Hz cycle, sim_world camera_info every
        # detector tick. Below nominal means the stack did not keep up (e.g. sim_world ~speed too high).
        self.rates = {"navigator": RateTracker(10.0), "detector": RateTracker(float(rospy.get_param("~detect_hz", 5.0)))}
        rospy.Subscriber("/cmd_vel_mux/input/nav_vel", Twist, lambda _m: self.rates["navigator"].tick(rospy.get_time()),
                         queue_size=50)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo,
                         lambda _m: self.rates["detector"].tick(rospy.get_time()), queue_size=50)
        rospy.on_shutdown(self.write_summary)

    def note(self, line):
        self.log.append({"t": round(rospy.get_time() - self.t_start, 1), "msg": line})
        rospy.loginfo("sim_monitor: %s", line)

    def truth_callback(self, msg):
        self.scorer.set_ground_truth(json.loads(msg.data)["placements"])

    def beliefs_callback(self, msg):
        objects = [(o.object_id, o.class_name, o.local_x, o.local_y) for o in msg.objects]
        for line in self.scorer.update_ledger(rospy.get_time(), objects):
            self.note(line)
        self.check_blockouts()

    def stop_callback(self, msg):
        name = EVENT_NAMES.get(msg.event, str(msg.event))
        line = self.scorer.add_stop_event(name, kind_name(msg.kind), msg.object_id, rospy.get_time(),
                                          robot_xy=(msg.x, msg.y), distance=msg.distance)
        if line:
            self.note(line)

    def runner_callback(self, msg):
        self.note(self.scorer.add_runner_event(msg.data))

    def rosout_callback(self, msg):
        if msg.name != self.navigator_name:
            return
        line = self.scorer.add_nav_log(msg.msg)
        if line:
            self.note(line)

    def result_callback(self, msg):
        outcome = OUTCOMES[msg.outcome] if msg.outcome < len(OUTCOMES) else str(msg.outcome)
        self.note(self.scorer.add_result(rospy.get_time(), msg.object_id, msg.class_name, outcome, msg.reason,
                                         window_start=msg.window_start or None, window_end=msg.window_end or None))

    def object_map_callback(self, msg):
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        info = msg.info

        def is_blocked(x, y):
            col = int((x - info.origin.position.x) / info.resolution)
            row = int((y - info.origin.position.y) / info.resolution)
            if not (0 <= row < info.height and 0 <= col < info.width):
                return False
            return grid[row, col] >= 50

        self.is_blocked = is_blocked
        self.check_blockouts()

    def check_blockouts(self):
        """The mapper only republishes /object_map on changes, so also check the latest one after each removal."""
        if self.is_blocked is None:
            return
        for line in self.scorer.update_blockouts(rospy.get_time(), self.is_blocked):
            self.note(line)

    def write_summary(self):
        self.check_blockouts()
        summary = self.scorer.summary(rospy.get_time())
        summary["scenario"] = self.scenario.name
        summary["run_s"] = round(rospy.get_time() - self.t_start, 1)
        summary["run_wall_s"] = round(time.time() - self.wall_start, 1)  # wall clock: speedup = run_s / run_wall_s
        summary["loop_rates"] = {name: r.summary() for name, r in self.rates.items()}
        summary["loop_rates_ok"] = all(r.ok() is not False for r in self.rates.values())
        if not summary["loop_rates_ok"]:
            rospy.logwarn("sim_monitor: loop rates below 90%% of nominal (%s): the stack did not keep up with "
                          "simulated time; lower sim_world's speed", summary["loop_rates"])
        checks = check_expectations(summary, self.scenario.expect)
        if checks:  # a sped-up run only counts if the stack kept up
            checks.append({"key": "loop_rates_ok", "op": "==", "expected": 1, "actual": int(summary["loop_rates_ok"]),
                           "ok": summary["loop_rates_ok"]})
            summary["expectations"] = checks
            summary["expectations_passed"] = all(c["ok"] for c in checks)
        text = json.dumps(summary, indent=2)
        rospy.loginfo("sim_monitor: run summary\n%s", text)
        print("\n===== sim_monitor summary (%s) =====\n%s\n" % (self.scenario.name, text), flush=True)
        if checks:
            lines = ["%s  %s %s %g (got %s)%s" % ("PASS" if c["ok"] else "FAIL", c["key"], c["op"], c["expected"],
                                                  c["actual"], " " + c["error"] if c.get("error") else "")
                     for c in checks]
            print("===== expectations: %s =====\n%s\n" % (
                "PASSED" if summary["expectations_passed"] else "FAILED", "\n".join(lines)), flush=True)
        if not self.results_dir:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "%s_%s.json" % (self.scenario.name, time.strftime("%Y%m%d_%H%M%S")))
        with open(path, "w") as f:
            json.dump({"summary": summary, "log": self.log, "removals": self.scorer.removals,
                       "observations": self.scorer.results, "ledger_added": self.scorer.added,
                       "stops": self.scorer.stop_log}, f, indent=2)
        print("sim_monitor: wrote %s" % path, flush=True)


if __name__ == "__main__":
    SimMonitor()
    rospy.spin()
