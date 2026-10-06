"""Simulated object detector: ground-truth objects in the camera's field of view and line of sight.

Mimics what detect_with_dist_osod.py publishes on /detected_objects (map-frame positions, metric width,
bounding box), with optional misses, position noise and false positives (all off by default).
Pure Python + numpy (no ROS).
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from mattbot_sim.kinematics import wrap_angle

TURN_RATE_THRESHOLD = 0.15  # rad/s; the real detector drops frames while turning faster than this
TURN_HOLD_S = 0.25  # ... and for this long afterwards


@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    fx: float = 554.3  # ~60 deg horizontal FOV, close to the Astra colour camera
    fy: float = 554.3
    cx: float = 319.5
    cy: float = 239.5
    max_range: float = 5.0  # the detector drops objects beyond 5 m
    min_range: float = 0.3
    object_height: float = 0.5  # for the (cosmetic) bounding box rows

    @property
    def hfov(self):
        return 2.0 * math.atan(self.width / (2.0 * self.fx))


@dataclass
class DetectorNoise:
    miss_prob: float = 0.0  # per object per frame
    pos_noise_std: float = 0.0  # m, isotropic, added to the reported position
    false_pos_rate: float = 0.0  # expected false detections per frame
    false_pos_classes: tuple = ("chair",)


@dataclass
class Detection:
    class_name: str
    x: float  # map frame
    y: float
    width: float
    probability: float
    x1: float  # bounding box, pixels
    y1: float
    x2: float
    y2: float
    object_id: str = ""  # ground-truth id ("" for false positives); not published


def visible_objects(world, cam_x, cam_y, cam_yaw, camera):
    """[(SimObject, range, bearing)] for present objects inside the FOV, within range and in line of sight."""
    out = []
    half_fov = 0.5 * camera.hfov
    for obj in world.present_objects():
        rng = math.hypot(obj.x - cam_x, obj.y - cam_y)
        if rng > camera.max_range or rng < camera.min_range:
            continue
        brg = wrap_angle(math.atan2(obj.y - cam_y, obj.x - cam_x) - cam_yaw)
        if abs(brg) > half_fov:
            continue
        if not world.line_of_sight(cam_x, cam_y, obj):
            continue
        out.append((obj, rng, brg))
    return out


def bounding_box(rng, brg, width, camera, cam_height):
    """Approximate pixel box of an upright object of ``width`` at (range, bearing), clipped to the image."""
    depth = max(rng * math.cos(brg), 1e-3)
    lateral = rng * math.sin(brg)  # left of the optical axis is positive
    u = camera.cx - camera.fx * lateral / depth
    half = camera.fx * 0.5 * width / depth
    v_top = camera.cy - camera.fy * (camera.object_height - cam_height) / depth
    v_bot = camera.cy + camera.fy * cam_height / depth
    clip_u = lambda v: float(min(max(v, 0.0), camera.width - 1))  # noqa: E731
    clip_v = lambda v: float(min(max(v, 0.0), camera.height - 1))  # noqa: E731
    return clip_u(u - half), clip_v(v_top), clip_u(u + half), clip_v(v_bot)


def detect(world, cam_x, cam_y, cam_yaw, cam_height, camera, noise=None, rng=None) -> List[Detection]:
    """One detector frame."""
    noise = noise or DetectorNoise()
    rng = rng or np.random.default_rng()
    dets = []
    for obj, r, b in visible_objects(world, cam_x, cam_y, cam_yaw, camera):
        if noise.miss_prob > 0.0 and rng.random() < noise.miss_prob:
            continue
        x, y = obj.x, obj.y
        if noise.pos_noise_std > 0.0:
            x += rng.normal(0.0, noise.pos_noise_std)
            y += rng.normal(0.0, noise.pos_noise_std)
        dets.append(Detection(obj.class_name, x, y, obj.width, 0.9, *bounding_box(r, b, obj.width, camera, cam_height),
                              object_id=obj.object_id))
    if noise.false_pos_rate > 0.0:
        for _ in range(rng.poisson(noise.false_pos_rate)):
            r = rng.uniform(1.0, camera.max_range)
            b = rng.uniform(-0.5 * camera.hfov, 0.5 * camera.hfov)
            cls = str(rng.choice(list(noise.false_pos_classes)))
            dets.append(Detection(cls, cam_x + r * math.cos(cam_yaw + b), cam_y + r * math.sin(cam_yaw + b), 0.5, 0.6,
                                  *bounding_box(r, b, 0.5, camera, cam_height)))
    return dets


class TurnGate:
    """Drops frames while the base turns, like detect_with_dist_osod.cmd_vel_callback."""

    def __init__(self, threshold=TURN_RATE_THRESHOLD, hold_s=TURN_HOLD_S):
        self.threshold = threshold
        self.hold_s = hold_s
        self.last_turn = -math.inf

    def update(self, w, now):
        if abs(w) > self.threshold:
            self.last_turn = now

    def turning(self, now):
        return now - self.last_turn < self.hold_s
