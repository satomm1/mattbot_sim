#!/usr/bin/env python3
"""Drives a sim run from the scenario file: patrol goals plus timed world events.

1. Waits until the navigator is localized (/localized, or /robot_mode back to IDLE after LOCALIZING2).
2. Sends the waypoints one at a time on /external_goal (as patrol.py does). A goal is done when the
   navigator is IDLE again within ~reach_tol_m of it; if it went IDLE short of the goal the same goal
   is re-sent (the navigator replans a duplicate goal while IDLE), and after ~max_retries it moves on.
3. Publishes each scenario event on /sim/event when its time comes (s after localization).
4. With a scenario duration (or ~duration_s > 0), shuts down when it is reached; launch this node with
   required="true" so the whole sim stops and sim_monitor writes its summary.

Objects marked known_from_peer are put in the ledger right after localization, as observations of the
scenario's peer robot on /ledger/observation_from_agent (mattbot_sim/peers.py).
Goal re-sends and give-ups are also published on /sim/runner_event (String) so sim_monitor can count
them: a re-send can hide a navigator that went IDLE without reaching the goal (e.g. after a detour).
"""

import math
import time

import rospy
import yaml
from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import Bool, Int32, String

from mattbot_sim.clock import wait_for_clock
from mattbot_sim.peers import peer_messages
from mattbot_sim.scenario import load_scenario, robot_id_from_env
from mattbot_sim.world import SimObject

IDLE, LOCALIZING2 = 0, 2


def event_yaml(event):
    e = {k: v for k, v in event.items() if k != "at"}
    if isinstance(e.get("add"), SimObject):
        o = e["add"]
        e["add"] = {"id": o.object_id, "class": o.class_name, "x": o.x, "y": o.y, "width": o.width}
    return yaml.safe_dump(e, default_flow_style=True).strip()


class ScenarioRunner:
    def __init__(self):
        rospy.init_node("scenario_runner")
        wait_for_clock()  # all times here are ROS time (simulated with /use_sim_time)
        self.scenario = load_scenario(rospy.get_param("/sim/scenario_file"), robot_id_from_env())
        duration = float(rospy.get_param("~duration_s", -1.0))
        self.duration = duration if duration >= 0.0 else self.scenario.duration
        self.reach_tol = float(rospy.get_param("~reach_tol_m", 0.5))
        self.idle_settle_s = float(rospy.get_param("~idle_settle_s", 1.5))
        self.pause_s = float(rospy.get_param("~pause_at_waypoint_s", 2.0))
        self.goal_timeout_s = float(rospy.get_param("~goal_timeout_s", 300.0))
        self.max_retries = int(rospy.get_param("~max_retries", 3))

        self.mode = None
        self.mode_since = rospy.get_time()
        self.prev_mode = None
        self.localized = False
        self.pose = None

        self.goal_pub = rospy.Publisher("/external_goal", Pose2D, queue_size=10)
        self.event_pub = rospy.Publisher("/sim/event", String, queue_size=10)
        self.runner_pub = rospy.Publisher("/sim/runner_event", String, queue_size=10)
        self.peer_pub = rospy.Publisher("/ledger/observation_from_agent", String, queue_size=50)
        rospy.Subscriber("/robot_mode", Int32, self.mode_callback, queue_size=10)
        rospy.Subscriber("/localized", Bool, self.localized_callback, queue_size=1)
        rospy.Subscriber("/sim/true_pose", PoseStamped, self.pose_callback, queue_size=1)

    def mode_callback(self, msg):
        if msg.data != self.mode:
            self.prev_mode, self.mode, self.mode_since = self.mode, msg.data, rospy.get_time()
            if msg.data == IDLE and self.prev_mode == LOCALIZING2:
                self.localized = True

    def localized_callback(self, msg):
        self.localized = self.localized or msg.data

    def pose_callback(self, msg):
        self.pose = (msg.pose.position.x, msg.pose.position.y)

    def idle_for(self):
        return rospy.get_time() - self.mode_since if self.mode == IDLE else 0.0

    def dist_to(self, wp):
        return math.hypot(self.pose[0] - wp.x, self.pose[1] - wp.y) if self.pose else math.inf

    def run(self):
        rospy.loginfo("scenario_runner: %s, waiting for the navigator to localize", self.scenario.name)
        while not rospy.is_shutdown() and not self.localized:
            rospy.sleep(0.5)
        t0 = rospy.get_time()
        epoch = float(rospy.get_param("/sim/clock_epoch", 0.0))
        if self.scenario.is_fleet and epoch > 0.0:
            # Fleet: every robot starts its patrol, and every event fires, at the same simulated time, so the
            # robots' separate worlds stay identical. fleet_start_s must cover localization and DDS discovery.
            t_start = epoch + self.scenario.fleet_start_s
            if t0 > t_start:
                rospy.logwarn("scenario_runner: localized %.0f s after the fleet start; raise fleet_start_s",
                              t0 - t_start)
            rospy.loginfo("scenario_runner: robot %s waiting for the fleet start (%.0f s)", self.scenario.robot_id,
                          max(t_start - t0, 0.0))
            while not rospy.is_shutdown() and rospy.get_time() < t_start:
                rospy.sleep(0.2)
            t0 = t_start
        self.seed_peer_objects(t0)
        rospy.loginfo("scenario_runner: localized; starting %d waypoints, %d events%s", len(self.scenario.waypoints),
                      len(self.scenario.events), ", %.0f s run" % self.duration if self.duration > 0 else "")

        events = list(self.scenario.events)
        waypoints = self.scenario.waypoints
        idx, sent_at, retries, laps = 0, None, 0, 0
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - t0
            while events and events[0]["at"] <= elapsed:
                e = events.pop(0)
                rospy.loginfo("scenario_runner: t=%.0f s event: %s", elapsed, event_yaml(e))
                self.event_pub.publish(String(data=event_yaml(e)))
            if self.duration > 0 and elapsed >= self.duration:
                rospy.loginfo("scenario_runner: duration %.0f s reached, ending the run", self.duration)
                rospy.signal_shutdown("scenario finished")
                break

            if waypoints and idx is not None:
                wp = waypoints[idx]
                if sent_at is None:
                    self.send(wp, idx)
                    sent_at = rospy.get_time()
                elif rospy.get_time() - sent_at > 3.0 and self.idle_for() > self.idle_settle_s:
                    if self.dist_to(wp) <= self.reach_tol or retries >= self.max_retries:
                        if self.dist_to(wp) > self.reach_tol:
                            rospy.logwarn("scenario_runner: giving up on waypoint %d after %d retries", idx, retries)
                            self.runner_event("give_up goal %d %.1f m short" % (idx, self.dist_to(wp)))
                        pause = self.scenario.waypoint_pauses[idx] if idx < len(self.scenario.waypoint_pauses) else None
                        rospy.sleep(self.pause_s if pause is None else pause)
                        idx, sent_at, retries = idx + 1, None, 0
                        if idx >= len(waypoints):
                            laps += 1
                            idx = 0 if self.scenario.loop else None
                            rospy.loginfo("scenario_runner: finished lap %d", laps)
                    else:
                        retries += 1
                        rospy.logwarn("scenario_runner: navigator idle %.1f m short of waypoint %d; re-sending (%d/%d)",
                                      self.dist_to(wp), idx, retries, self.max_retries)
                        self.runner_event("resend goal %d %.1f m short" % (idx, self.dist_to(wp)))
                        sent_at = None
                elif rospy.get_time() - sent_at > self.goal_timeout_s:
                    rospy.logwarn("scenario_runner: waypoint %d timed out; re-sending", idx)
                    self.runner_event("resend goal %d timed out" % idx)
                    sent_at = None
            rate.sleep()

    def runner_event(self, text):
        self.runner_pub.publish(String(data=text))

    def seed_peer_objects(self, t0):
        ids = set(self.scenario.known_from_peer)
        objects = [o for o in self.scenario.objects if o.object_id in ids]
        if not objects:
            return
        deadline = rospy.get_time() + 10.0
        while not rospy.is_shutdown() and self.peer_pub.get_num_connections() == 0 and rospy.get_time() < deadline:
            rospy.sleep(0.2)  # messages published before the ledger connects are lost
        if self.peer_pub.get_num_connections() == 0:
            rospy.logwarn("scenario_runner: no subscriber on /ledger/observation_from_agent; is observation_ledger running?")
        for o, msg in zip(objects, peer_messages(objects, self.scenario.peer_id, int(t0), rospy.get_time())):
            rospy.loginfo("scenario_runner: peer %d reports %s (%s) at (%.2f, %.2f)",
                          self.scenario.peer_id, o.object_id, o.class_name, o.x, o.y)
            self.peer_pub.publish(String(data=msg))
            rospy.sleep(0.05)

    def send(self, wp, idx):
        rospy.loginfo("scenario_runner: goal %d -> (%.2f, %.2f, %.2f)", idx, wp.x, wp.y, wp.theta)
        self.goal_pub.publish(Pose2D(x=wp.x, y=wp.y, theta=wp.theta))


if __name__ == "__main__":
    ScenarioRunner().run()
