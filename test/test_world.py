"""Unit tests for mattbot_sim/world.py (no ROS needed)."""

import json
import math

import numpy as np
import pytest

from mattbot_sim.world import GridMap, SimObject, World, load_json_map, ray_circle


def box_map(size_m=10.0, res=0.1):
    """Free square room with a one-cell wall all around; origin at (0, 0)."""
    n = int(size_m / res)
    occ = np.zeros((n, n), dtype=np.int16)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    return GridMap(occ, res, 0.0, 0.0)


def test_raycast_hits_wall():
    world = World(box_map())
    r = world.raycast(5.0, 5.0, [0.0, math.pi / 2, math.pi], 20.0)
    assert np.allclose(r, [4.9, 4.9, 4.9], atol=0.1)


def test_raycast_beyond_max_range_is_inf():
    world = World(box_map())
    assert np.isinf(world.raycast(5.0, 5.0, [0.0], 2.0)[0])


def test_unknown_cells_block():
    grid = box_map()
    grid.occupancy[:, 70] = -1
    world = World(GridMap(grid.occupancy, grid.resolution, 0.0, 0.0))
    assert world.raycast(5.0, 5.0, [0.0], 20.0)[0] == pytest.approx(2.0, abs=0.1)


def test_raycast_hits_object_and_respects_exclude():
    obj = SimObject("c", "chair", 7.0, 5.0, width=0.5)
    world = World(box_map(), [obj])
    assert world.raycast(5.0, 5.0, [0.0], 20.0)[0] == pytest.approx(1.75, abs=1e-6)
    assert world.raycast(5.0, 5.0, [0.0], 20.0, exclude=("c",))[0] == pytest.approx(4.9, abs=0.1)


def test_ray_circle_miss_and_behind():
    assert np.isinf(ray_circle(0, 0, [math.pi / 2], 2.0, 0.0, 0.5)[0])
    assert np.isinf(ray_circle(0, 0, [math.pi], 2.0, 0.0, 0.5)[0])
    assert ray_circle(2.0, 0.0, [0.0], 2.0, 0.0, 0.5)[0] == 0.0


def test_line_of_sight_blocked_by_other_object():
    target = SimObject("t", "cone", 8.0, 5.0, width=0.3)
    blocker = SimObject("b", "bench", 6.5, 5.0, width=0.6)
    world = World(box_map(), [target, blocker])
    assert not world.line_of_sight(5.0, 5.0, target)
    world.apply_event({"remove": "b"}, now=1.0)
    assert world.line_of_sight(5.0, 5.0, target)


def test_line_of_sight_blocked_by_wall():
    grid = box_map()
    grid.occupancy[20:80, 60] = 100  # wall at x = 6
    world = World(GridMap(grid.occupancy, grid.resolution, 0.0, 0.0), [SimObject("t", "cone", 8.0, 5.0)])
    assert not world.line_of_sight(5.0, 5.0, world.objects["t"])


def test_object_against_wall_is_visible():
    """An object touching a wall behind it is still in view (the ray stops at its near side)."""
    obj = SimObject("t", "chair", 9.6, 5.0, width=0.5)  # wall cells start at x = 9.9
    world = World(box_map(), [obj])
    assert world.line_of_sight(5.0, 5.0, obj)


def test_collides():
    world = World(box_map(), [SimObject("c", "chair", 5.0, 5.0, width=0.5)])
    assert world.collides(5.3, 5.0, 0.2)
    assert not world.collides(6.0, 5.0, 0.2)
    assert world.collides(0.15, 5.0, 0.2)  # wall


def test_events_track_placements():
    world = World(box_map(), [SimObject("c", "chair", 5.0, 5.0)], now=0.0)
    world.apply_event({"remove": "c"}, now=10.0)
    world.apply_event({"restore": "c"}, now=20.0)
    world.apply_event({"move": "c", "x": 6.0, "y": 6.0}, now=30.0)
    world.apply_event({"add": SimObject("k", "cone", 3.0, 3.0)}, now=40.0)
    spans = [(p.object_id, p.x, p.t_start, p.t_end) for p in world.placements]
    assert spans == [("c", 5.0, 0.0, 10.0), ("c", 5.0, 20.0, 30.0), ("c", 6.0, 30.0, None), ("k", 3.0, 40.0, None)]
    with pytest.raises(ValueError):
        world.apply_event({"remove": "nope"}, now=50.0)


def test_load_json_map(tmp_path):
    occ = [0, 100, -1, 0, 0, 0]
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"data": {"map": {"width": 3, "height": 2, "resolution": 0.05,
                                                  "origin_x": 1.0, "origin_y": 2.0, "occupancy": occ}}}))
    grid = load_json_map(str(path))
    assert grid.occupancy.shape == (2, 3)
    assert grid.occupancy[0, 1] == 100  # row 0 = first width entries (origin row)
    assert grid.is_free(1.01, 2.01) and not grid.is_free(1.06, 2.01)
