#!/usr/bin/env python3
"""2D world simulator: stands in for the robot hardware, localization and the object detector.

Replaces: mcu_comms.py   /cmd_vel -> /odom + TF odom -> base_footprint (unicycle model)
          amcl           TF map -> odom, perfect localization (map -> base_footprint is ground truth)
          entry_exit.py  latched /map, /map_mod, /map_metadata from the scenario's map
          camera + detect_with_dist_osod.py
                         /detected_objects (perfect detections of ground-truth objects in view)
                         /camera/color/camera_info (synthetic intrinsics; FOV for the evaluator)
          rplidar        /scan (only with ~scan, for RViz; nothing in the stack reads it)
Also:     /initialpose   latched start pose, so the navigator leaves WAITING_FOR_INIT
          /sim/event     std_msgs/String YAML world event, e.g. "remove: chair_1" (see scenario.py)
          /sim/ground_truth  latched JSON placements, for sim_monitor
          /sim/objects   MarkerArray (green = present, grey = removed), /sim/true_pose

Simulated time: with /use_sim_time, sim_world is the clock. A wall-paced thread advances /clock by a
fixed physics step (1/~rate_hz) every step / ~speed wall seconds, starting at the current Unix time
(so stamps, ledger sessions and logs still look like wall time). Everything else in the stack runs on
ROS time and follows it; ~speed > 1 runs the scenario faster than real time as long as the nodes keep
up (sim_monitor checks the navigator and detector rates). Without /use_sim_time it runs on the wall
clock as before.
"""

import json
import math
import os
import threading
import time

import numpy as np
import rosnode
import rospkg
import rospy
import tf2_ros
import xml.etree.ElementTree as ET
import yaml
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

from mattbot_sim.kinematics import Limits, Pose2D, clamp, compose, inverse, step
from mattbot_sim.perception import CameraModel, DetectorNoise, TurnGate, detect
from mattbot_sim.scenario import load_scenario, parse_event
from mattbot_sim.world import World, load_maps

CMD_TIMEOUT_S = 0.5  # stop if /cmd_vel goes quiet (as the MCU would)
HARDWARE_NODES = ("/mcu_comms_node", "/rplidar_node", "/camera/camera_nodelet_manager")


def quaternion_msg(yaw):
    q = quaternion_from_euler(0.0, 0.0, yaw)
    return q[0], q[1], q[2], q[3]


def timer_callback(fn):
    """Timer callbacks may still fire while rospy shuts down; drop those instead of raising."""
    def wrapper(self, event):
        if rospy.is_shutdown():
            return
        try:
            fn(self, event)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise
    return wrapper


def frame_offset(urdf_path, child, root="base_footprint"):
    """Sum of fixed-joint xyz offsets from root to child (publish_transforms.py also ignores rpy)."""
    joints = {j.find("child").get("link"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    xyz = np.zeros(3)
    link = child
    while link != root:
        if link not in joints:
            raise ValueError("%s: no joint chain from %s to %s" % (urdf_path, root, child))
        joint = joints[link]
        origin = joint.find("origin")
        if origin is not None:
            xyz += np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        link = joint.find("parent").get("link")
    return xyz


def grid_msg(grid, stamp):
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = grid.resolution
    msg.info.width = grid.width
    msg.info.height = grid.height
    msg.info.origin.position.x = grid.origin_x
    msg.info.origin.position.y = grid.origin_y
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.occupancy.astype(np.int8).ravel().tolist()
    return msg


class SimWorld:
    def __init__(self):
        rospy.init_node("sim_world")
        self.check_no_hardware()
        self.sim_time = bool(rospy.get_param("/use_sim_time", False))
        self.speed = float(rospy.get_param("~speed", 1.0))
        self.dt = 1.0 / float(rospy.get_param("~rate_hz", 50.0))
        if self.sim_time:
            if self.speed <= 0.0:
                raise SystemExit("sim_world: ~speed must be > 0")
            # Start the clock before anything reads ROS time (it is 0 until the first /clock)
            self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
            self.t_sim = time.time()  # wall clock: the sim clock starts at the current Unix time
            while not rospy.is_shutdown() and rospy.get_time() <= 0.0:
                self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(self.t_sim)))
                time.sleep(0.05)  # wall clock: waiting for our own first /clock to arrive

        rospack = rospkg.RosPack()
        self.scenario = load_scenario(rospy.get_param("/sim/scenario_file"))
        map_json_dir = os.path.join(rospack.get_path("mattbot_mcl"), "map_json")
        self.grid, self.grid_mod = load_maps(self.scenario.map, map_json_dir)
        self.world = World(self.grid, self.scenario.objects, now=rospy.get_time())

        urdf = rospy.get_param("~urdf", os.path.join(rospack.get_path("mattbot_bringup"), "urdf", "robot_tf_short.urdf"))
        self.cam_offset = frame_offset(urdf, "camera_link")
        self.laser_offset = frame_offset(urdf, "laser_frame")
        self.robot_radius = float(rospy.get_param("~robot_radius", 0.2))
        self.limits = Limits(float(rospy.get_param("~v_max", 0.7)), float(rospy.get_param("~w_max", 2.5)))
        self.odom_noise = float(rospy.get_param("~odom_noise", 0.0))  # relative std on v and w
        self.camera = CameraModel(max_range=float(rospy.get_param("~detect_max_range", 5.0)))
        self.noise = DetectorNoise(
            miss_prob=float(rospy.get_param("~miss_prob", 0.0)),
            pos_noise_std=float(rospy.get_param("~pos_noise_std", 0.0)),
            false_pos_rate=float(rospy.get_param("~false_pos_rate", 0.0)),
        )
        self.rng = np.random.default_rng(int(rospy.get_param("~seed", 0)))
        self.robot_id = int(os.environ.get("ROBOT_ID", "0"))

        if self.world.collides(self.scenario.start.x, self.scenario.start.y, self.robot_radius):
            rospy.logwarn("sim_world: start pose (%.2f, %.2f) is not free", self.scenario.start.x, self.scenario.start.y)
        for obj in self.scenario.objects:
            if not self.grid.is_free(obj.x, obj.y):
                rospy.logwarn("sim_world: object %s at (%.2f, %.2f) is not on a free map cell", obj.object_id, obj.x, obj.y)

        self.lock = threading.Lock()
        self.truth = Pose2D(self.scenario.start.x, self.scenario.start.y, self.scenario.start.theta)
        self.odom = Pose2D()  # odometry starts at zero like the MCU's
        self.odom_origin = self.truth  # map pose of the odom frame (fixed: perfect localization)
        self.v = self.w = 0.0
        self.last_cmd = -math.inf
        self.turn_gate = TurnGate()
        self.last_step = rospy.get_time()

        latched = dict(queue_size=1, latch=True)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.det_pub = rospy.Publisher("/detected_objects", DetectedObjectArray, queue_size=5)
        self.info_pub = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=1)
        self.scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=1)
        self.truth_pub = rospy.Publisher("/sim/true_pose", PoseStamped, queue_size=1)
        self.markers_pub = rospy.Publisher("/sim/objects", MarkerArray, **latched)
        self.gt_pub = rospy.Publisher("/sim/ground_truth", String, **latched)
        map_pub = rospy.Publisher("/map", OccupancyGrid, **latched)
        map_mod_pub = rospy.Publisher("/map_mod", OccupancyGrid, **latched)
        map_md_pub = rospy.Publisher("/map_metadata", MapMetaData, **latched)
        init_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, **latched)

        stamp = rospy.Time.now()
        map_msg = grid_msg(self.grid, stamp)
        map_pub.publish(map_msg)
        map_mod_pub.publish(grid_msg(self.grid_mod, stamp))
        map_md_pub.publish(map_msg.info)
        init_pub.publish(self.initial_pose_msg())
        self.publish_world_state()

        rospy.Subscriber("/cmd_vel", Twist, self.cmd_callback, queue_size=1)
        rospy.Subscriber("/sim/event", String, self.event_callback, queue_size=10)

        if self.sim_time:
            threading.Thread(target=self.clock_loop, name="sim_clock", daemon=True).start()
        else:
            rospy.Timer(rospy.Duration(self.dt), self.physics_timer)
        rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~detect_hz", 5.0))), self.detect_step)
        rospy.Timer(rospy.Duration(0.1), self.publish_truth)
        rospy.Timer(rospy.Duration(1.0), self.world_state_timer)
        if rospy.get_param("~scan", False):
            rospy.Timer(rospy.Duration(0.1), self.scan_step)
        rospy.loginfo("sim_world: scenario %s, map %s (%dx%d), %d objects, start (%.2f, %.2f, %.2f)%s",
                      self.scenario.name, self.scenario.map, self.grid.width, self.grid.height,
                      len(self.scenario.objects), self.truth.x, self.truth.y, self.truth.theta,
                      ", simulated time at %gx" % self.speed if self.sim_time else ", wall clock")

    @staticmethod
    def check_no_hardware():
        """Refuse to run on a ROS master that has the real robot drivers (our /cmd_vel would drive it)."""
        try:
            nodes = set(rosnode.get_node_names())
        except rosnode.ROSNodeIOException:
            return
        clash = sorted(nodes.intersection(HARDWARE_NODES))
        if clash:
            rospy.logfatal("sim_world: real robot nodes %s are on this ROS master; run the sim on its own master "
                           "(e.g. ROS_MASTER_URI=http://localhost:11411)", clash)
            raise SystemExit(1)

    def initial_pose_msg(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.truth.x, self.truth.y
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = \
            quaternion_msg(self.truth.theta)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.01
        msg.pose.covariance[35] = 0.01
        return msg

    # ---------- Inputs ----------

    def cmd_callback(self, msg):
        with self.lock:
            self.v = clamp(msg.linear.x, -self.limits.v_max, self.limits.v_max)
            self.w = clamp(msg.angular.z, -self.limits.w_max, self.limits.w_max)
            self.last_cmd = rospy.get_time()
            self.turn_gate.update(msg.angular.z, self.last_cmd)

    def event_callback(self, msg):
        try:
            event = parse_event(yaml.safe_load(msg.data))
            with self.lock:
                desc = self.world.apply_event(event, rospy.get_time())
        except (ValueError, yaml.YAMLError) as e:
            rospy.logerr("sim_world: bad /sim/event %r: %s", msg.data, e)
            return
        rospy.loginfo("sim_world: %s", desc)
        self.publish_world_state()

    # ---------- Motion ----------

    def clock_loop(self):
        """Simulated time: fixed physics steps, /clock published after each, paced to ~speed x wall time."""
        wall_next = time.time()  # wall clock: pacing
        report_sim, report_wall = self.t_sim, time.time()  # wall clock: achieved-speed report
        while not rospy.is_shutdown():
            self.t_sim += self.dt
            try:
                self.physics_step(self.t_sim, self.dt)
                self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(self.t_sim)))
            except rospy.ROSException:
                if rospy.core.is_shutdown_requested():
                    return  # topics close before is_shutdown() turns true
                raise
            wall_next += self.dt / self.speed
            delay = wall_next - time.time()  # wall clock: pacing
            if delay > 0.0:
                time.sleep(delay)  # wall clock: pacing
            elif delay < -1.0:
                wall_next = time.time()  # wall clock: fell behind; do not try to catch up in a burst
            if self.t_sim - report_sim >= 30.0:
                wall = time.time()  # wall clock: achieved-speed report
                rospy.loginfo("sim_world: simulated time at %.2fx (asked %gx)",
                              (self.t_sim - report_sim) / max(wall - report_wall, 1e-6), self.speed)
                report_sim, report_wall = self.t_sim, wall

    @timer_callback
    def physics_timer(self, _event):
        now = rospy.get_time()
        dt = min(now - self.last_step, 0.1)
        self.last_step = now
        self.physics_step(now, dt)

    def physics_step(self, now, dt):
        with self.lock:
            if now - self.last_cmd > CMD_TIMEOUT_S:
                self.v = self.w = 0.0
            v, w = self.v, self.w
            nxt = step(self.truth, v, w, dt)
            if self.world.collides(nxt.x, nxt.y, self.robot_radius) and not self.world.collides(
                    self.truth.x, self.truth.y, self.robot_radius):
                # Blocked: wheels spin but the base does not move (the navigator's stall detection sees this)
                nxt = Pose2D(self.truth.x, self.truth.y, step(self.truth, 0.0, w, dt).theta)
                v_actual = 0.0
                rospy.logwarn_throttle(5.0, "sim_world: robot is in contact with an obstacle")
            else:
                v_actual = v
            self.truth = nxt
            if self.odom_noise > 0.0:
                v_meas = v_actual * (1.0 + self.rng.normal(0.0, self.odom_noise))
                w_meas = w * (1.0 + self.rng.normal(0.0, self.odom_noise))
                self.odom = step(self.odom, v_meas, w_meas, dt)
                # Perfect localization: move map -> odom so map -> base_footprint stays the truth
                self.odom_origin = compose(self.truth, inverse(self.odom))
            else:
                self.odom = compose(inverse(self.odom_origin), self.truth)
            odom, origin = self.odom, self.odom_origin

        stamp = rospy.Time.from_sec(now)
        self.tf_pub.sendTransform([
            self.transform(stamp, "map", "odom", origin),
            self.transform(stamp, "odom", "base_footprint", odom),
        ])
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_footprint"
        msg.pose.pose.position.x, msg.pose.pose.position.y = odom.x, odom.y
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = quaternion_msg(odom.theta)
        msg.twist.twist.linear.x = v_actual
        msg.twist.twist.angular.z = w
        self.odom_pub.publish(msg)

    @staticmethod
    def transform(stamp, parent, child, pose):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y = pose.x, pose.y
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = quaternion_msg(pose.theta)
        return t

    def sensor_pose(self, offset):
        with self.lock:
            truth = self.truth
        p = compose(truth, Pose2D(offset[0], offset[1], 0.0))
        return p.x, p.y, truth.theta, offset[2]

    # ---------- Perception ----------

    @timer_callback
    def detect_step(self, _event):
        now = rospy.get_time()
        stamp = rospy.Time.now()
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_link"
        info.width, info.height = self.camera.width, self.camera.height
        info.K = [self.camera.fx, 0.0, self.camera.cx, 0.0, self.camera.fy, self.camera.cy, 0.0, 0.0, 1.0]
        info.P = [self.camera.fx, 0.0, self.camera.cx, 0.0, 0.0, self.camera.fy, self.camera.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        info.D = [0.0] * 5
        self.info_pub.publish(info)

        with self.lock:
            if self.turn_gate.turning(now):
                return  # the real detector publishes nothing while turning
        x, y, yaw, height = self.sensor_pose(self.cam_offset)
        with self.lock:
            dets = detect(self.world, x, y, yaw, height, self.camera, self.noise, self.rng)
        msg = DetectedObjectArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.sending_agent = self.robot_id
        for d in dets:
            obj = DetectedObject(class_name=d.class_name, probability=d.probability, width=d.width,
                                 x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2)
            obj.pose = Pose()
            obj.pose.position.x, obj.pose.position.y = d.x, d.y
            obj.pose.orientation.w = 1.0
            msg.objects.append(obj)
        self.det_pub.publish(msg)

    @timer_callback
    def scan_step(self, _event):
        x, y, yaw, _z = self.sensor_pose(self.laser_offset)
        n = 720
        angles = -math.pi + np.arange(n) * (2.0 * math.pi / n)
        with self.lock:
            ranges = self.world.raycast(x, y, yaw + angles, 12.0)
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "laser_frame"
        msg.angle_min, msg.angle_increment = -math.pi, 2.0 * math.pi / n
        msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
        msg.scan_time, msg.time_increment = 0.1, 0.1 / n
        msg.range_min, msg.range_max = 0.15, 12.0
        msg.ranges = np.where(np.isfinite(ranges), ranges, np.inf).astype(np.float32).tolist()
        self.scan_pub.publish(msg)

    # ---------- Ground truth ----------

    @timer_callback
    def publish_truth(self, _event):
        with self.lock:
            truth = self.truth
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y = truth.x, truth.y
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = quaternion_msg(truth.theta)
        self.truth_pub.publish(msg)

    @timer_callback
    def world_state_timer(self, _event):
        self.publish_world_state()

    def publish_world_state(self):
        with self.lock:
            placements = [p.to_dict() for p in self.world.placements]
            objects = list(self.world.objects.values())
        self.gt_pub.publish(String(data=json.dumps({"placements": placements})))
        markers = MarkerArray()
        stamp = rospy.Time.now()
        for i, obj in enumerate(sorted(objects, key=lambda o: o.object_id)):
            body = Marker(type=Marker.CYLINDER, action=Marker.ADD, ns="sim_objects", id=2 * i)
            body.header.frame_id, body.header.stamp = "map", stamp
            body.pose.position.x, body.pose.position.y, body.pose.position.z = obj.x, obj.y, 0.25
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = obj.width
            body.scale.z = 0.5
            body.color.r, body.color.g, body.color.b, body.color.a = (0.1, 0.8, 0.2, 0.9) if obj.present else (0.5, 0.5, 0.5, 0.3)
            label = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns="sim_labels", id=2 * i + 1)
            label.header.frame_id, label.header.stamp = "map", stamp
            label.pose.position.x, label.pose.position.y, label.pose.position.z = obj.x, obj.y, 0.8
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = obj.object_id + ("" if obj.present else " (removed)")
            markers.markers.extend([body, label])
        self.markers_pub.publish(markers)


if __name__ == "__main__":
    SimWorld()
    rospy.spin()
