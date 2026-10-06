"""Unicycle kinematics for the simulated differential-drive base (pure Python, no ROS)."""

import math
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class Limits:
    v_max: float = 0.7  # m/s
    w_max: float = 2.5  # rad/s


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def step(pose, v, w, dt):
    """Exact unicycle integration over dt with constant (v, w)."""
    if abs(w) < 1e-6:
        x = pose.x + v * dt * math.cos(pose.theta)
        y = pose.y + v * dt * math.sin(pose.theta)
    else:
        th1 = pose.theta + w * dt
        x = pose.x + v / w * (math.sin(th1) - math.sin(pose.theta))
        y = pose.y - v / w * (math.cos(th1) - math.cos(pose.theta))
    return Pose2D(x, y, wrap_angle(pose.theta + w * dt))


def compose(a, b):
    """a (+) b: pose b expressed in frame a, returned in a's parent frame."""
    c, s = math.cos(a.theta), math.sin(a.theta)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap_angle(a.theta + b.theta))


def inverse(a):
    c, s = math.cos(a.theta), math.sin(a.theta)
    return Pose2D(-c * a.x - s * a.y, s * a.x - c * a.y, wrap_angle(-a.theta))
