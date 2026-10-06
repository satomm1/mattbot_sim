"""Unit tests for mattbot_sim/perception.py (no ROS needed)."""

import math

import numpy as np

from mattbot_sim.perception import CameraModel, DetectorNoise, TurnGate, detect, visible_objects
from mattbot_sim.world import SimObject, World
from test_world import box_map

CAM = CameraModel()


def ids(world, x=2.0, y=5.0, yaw=0.0):
    return sorted(o.object_id for o, _r, _b in visible_objects(world, x, y, yaw, CAM))


def test_fov_and_range():
    world = World(box_map(), [
        SimObject("ahead", "chair", 5.0, 5.0),
        SimObject("behind", "chair", 1.0, 5.0),
        SimObject("side", "chair", 2.0, 8.0),  # 90 deg left
        SimObject("far", "chair", 8.0, 5.5),  # 6 m away
    ])
    assert ids(world) == ["ahead"]
    assert ids(world, yaw=math.pi / 2) == ["side"]


def test_edge_of_fov():
    half = 0.5 * CAM.hfov
    inside = SimObject("in", "cone", 2.0 + 3 * math.cos(half - 0.02), 5.0 + 3 * math.sin(half - 0.02))
    outside = SimObject("out", "cone", 2.0 + 3 * math.cos(-half - 0.02), 5.0 + 3 * math.sin(-half - 0.02))
    assert ids(World(box_map(), [inside, outside])) == ["in"]


def test_occlusion_and_removal():
    world = World(box_map(), [SimObject("t", "cone", 6.0, 5.0, 0.3), SimObject("b", "bench", 4.0, 5.0, 0.6)])
    assert ids(world) == ["b"]
    world.apply_event({"remove": "b"}, now=1.0)
    assert ids(world) == ["t"]
    world.apply_event({"remove": "t"}, now=2.0)
    assert ids(world) == []


def test_detection_fields():
    world = World(box_map(), [SimObject("c", "chair", 5.0, 5.0, 0.5)])
    (d,) = detect(world, 2.0, 5.0, 0.0, 0.2, CAM)
    assert (d.class_name, d.x, d.y, d.width, d.object_id) == ("chair", 5.0, 5.0, 0.5, "c")
    assert d.x1 < CAM.cx < d.x2  # centred in the image
    assert abs((d.x2 - d.x1) - CAM.fx * 0.5 / 3.0) < 1e-6  # width 0.5 m at 3 m depth


def test_noise_misses_and_false_positives():
    world = World(box_map(), [SimObject("c", "chair", 5.0, 5.0)])
    rng = np.random.default_rng(1)
    assert detect(world, 2.0, 5.0, 0.0, 0.2, CAM, DetectorNoise(miss_prob=1.0), rng) == []
    many = [len(detect(world, 2.0, 5.0, 0.0, 0.2, CAM, DetectorNoise(false_pos_rate=1.0), rng)) for _ in range(200)]
    assert 1.5 < np.mean(many) < 2.5  # 1 real + ~1 false per frame


def test_turn_gate():
    gate = TurnGate()
    gate.update(0.1, now=0.0)
    assert not gate.turning(0.0)
    gate.update(0.5, now=1.0)
    assert gate.turning(1.1)
    gate.update(0.0, now=1.2)
    assert gate.turning(1.2) and not gate.turning(1.3)
