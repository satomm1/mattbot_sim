"""Unit tests for mattbot_sim/kinematics.py (no ROS needed)."""

import math

import pytest

from mattbot_sim.kinematics import Pose2D, compose, inverse, step


def test_straight_line():
    p = step(Pose2D(0.0, 0.0, math.pi / 2), 0.5, 0.0, 2.0)
    assert (p.x, p.y) == pytest.approx((0.0, 1.0), abs=1e-9)


def test_full_circle_returns_home():
    p = Pose2D(1.0, 2.0, 0.3)
    for _ in range(100):
        p = step(p, 0.4, 2 * math.pi / 10.0, 0.1)  # one turn in 10 s
    assert (p.x, p.y, p.theta) == pytest.approx((1.0, 2.0, 0.3), abs=1e-6)


def test_turn_in_place():
    p = step(Pose2D(1.0, 1.0, 0.0), 0.0, 1.0, 0.5)
    assert (p.x, p.y, p.theta) == pytest.approx((1.0, 1.0, 0.5))


def test_compose_inverse():
    a = Pose2D(1.0, -2.0, 0.7)
    b = Pose2D(0.3, 0.4, -1.2)
    c = compose(inverse(a), compose(a, b))
    assert (c.x, c.y, c.theta) == pytest.approx((b.x, b.y, b.theta))
