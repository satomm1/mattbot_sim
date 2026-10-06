#!/usr/bin/env python3
"""Scores a sim run: compares the ledger and the observation outcomes with the ground truth.

Inputs: /sim/ground_truth (sim_world), /object_beliefs (active ledger objects), /observation/events,
        /observation/results, /object_map (occupancy_grid_mapper's object blockouts)
Logs each scored event and, on shutdown, prints a summary and writes it to
~results_dir/<scenario>_<time>.json (summary + event log).
"""

import json
import os
import time

import numpy as np
import rospy
from mattbot_dds.msg import ObjectBeliefArray, ObservationEvent, ObservationResult
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

from mattbot_sim.scenario import load_scenario
from mattbot_sim.scoring import OUTCOMES, RunScorer

EVENT_NAMES = {ObservationEvent.STARTED: "STARTED", ObservationEvent.ENDED: "ENDED", ObservationEvent.ABORTED: "ABORTED"}


class SimMonitor:
    def __init__(self):
        rospy.init_node("sim_monitor")
        self.scenario = load_scenario(rospy.get_param("/sim/scenario_file"))
        self.results_dir = rospy.get_param("~results_dir", "")
        self.scorer = RunScorer(match_radius=float(rospy.get_param("~match_radius_m", 1.0)))
        self.log = []
        self.t_start = time.time()
        self.is_blocked = None  # lookup into the latest /object_map

        rospy.Subscriber("/sim/ground_truth", String, self.truth_callback, queue_size=10)
        rospy.Subscriber("/object_beliefs", ObjectBeliefArray, self.beliefs_callback, queue_size=10)
        rospy.Subscriber("/observation/events", ObservationEvent, self.stop_callback, queue_size=50)
        rospy.Subscriber("/observation/results", ObservationResult, self.result_callback, queue_size=50)
        rospy.Subscriber("/object_map", OccupancyGrid, self.object_map_callback, queue_size=1)
        rospy.on_shutdown(self.write_summary)

    def note(self, line):
        self.log.append({"t": round(time.time() - self.t_start, 1), "msg": line})
        rospy.loginfo("sim_monitor: %s", line)

    def truth_callback(self, msg):
        self.scorer.set_ground_truth(json.loads(msg.data)["placements"])

    def beliefs_callback(self, msg):
        objects = [(o.object_id, o.class_name, o.local_x, o.local_y) for o in msg.objects]
        for line in self.scorer.update_ledger(time.time(), objects):
            self.note(line)
        self.check_blockouts()

    def stop_callback(self, msg):
        name = EVENT_NAMES.get(msg.event, str(msg.event))
        self.scorer.add_stop_event(name)
        if msg.event == ObservationEvent.STARTED:
            self.note("observation stop STARTED for %s at %.1f m" % (msg.object_id, msg.distance))
        elif msg.event == ObservationEvent.ABORTED:
            self.note("observation stop ABORTED for %s" % msg.object_id)

    def result_callback(self, msg):
        outcome = OUTCOMES[msg.outcome] if msg.outcome < len(OUTCOMES) else str(msg.outcome)
        self.note(self.scorer.add_result(time.time(), msg.object_id, msg.class_name, outcome, msg.reason,
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
        for line in self.scorer.update_blockouts(time.time(), self.is_blocked):
            self.note(line)

    def write_summary(self):
        self.check_blockouts()
        summary = self.scorer.summary(time.time())
        summary["scenario"] = self.scenario.name
        summary["run_s"] = round(time.time() - self.t_start, 1)
        text = json.dumps(summary, indent=2)
        rospy.loginfo("sim_monitor: run summary\n%s", text)
        print("\n===== sim_monitor summary (%s) =====\n%s\n" % (self.scenario.name, text), flush=True)
        if not self.results_dir:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "%s_%s.json" % (self.scenario.name, time.strftime("%Y%m%d_%H%M%S")))
        with open(path, "w") as f:
            json.dump({"summary": summary, "log": self.log, "removals": self.scorer.removals,
                       "observations": self.scorer.results, "ledger_added": self.scorer.added}, f, indent=2)
        print("sim_monitor: wrote %s" % path, flush=True)


if __name__ == "__main__":
    SimMonitor()
    rospy.spin()
