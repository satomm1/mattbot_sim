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


def test_thin_occluder_does_not_hide_a_wide_object():
    """A one-cell speck on the centre line (SLAM noise, a table leg) leaves most of a 1 m chair in view."""
    grid = box_map()
    grid.occupancy[50, 65] = 100  # one 0.1 m cell at (6.55, 5.05), on the line from (5, 5) to the chair
    world = World(GridMap(grid.occupancy, grid.resolution, 0.0, 0.0), [SimObject("t", "chair", 8.0, 5.0, width=1.0)])
    assert world.line_of_sight(5.0, 5.0, world.objects["t"])


def test_occluder_covering_most_of_the_object_hides_it():
    target = SimObject("t", "chair", 8.0, 5.0, width=1.0)
    world = World(box_map(), [target, SimObject("b", "bench", 6.5, 5.0, width=0.6)])  # covers > half the chair
    assert not world.line_of_sight(5.0, 5.0, target)
    edge = World(box_map(), [target, SimObject("b", "cone", 6.5, 5.4, width=0.2)])  # clips only one edge
    assert edge.line_of_sight(5.0, 5.0, target)


def test_camera_and_planner_use_the_same_visibility_rule():
    from mattbot_sim.fast import imports  # noqa: F401  (library paths)
    from navigation_utils import viewpoints
    assert tuple(World.LOS_OFFSETS) == tuple(viewpoints.LOS_OFFSETS)
    assert World.LOS_MIN_CLEAR == viewpoints.LOS_MIN_CLEAR


def test_planner_viewpoints_agree_with_the_camera():
    """Every cell the planner's width viewshed keeps, the sim camera sees the object from, anywhere within the
    pose tolerance of it; and the planner keeps most of the cells the camera sees from."""
    from mattbot_sim.fast import imports  # noqa: F401
    from navigation_utils.viewpoints import LOS_POSE_TOLERANCE_M, compute_viewshed
    rng = np.random.default_rng(0)
    occ = np.zeros((80, 120), dtype=np.int16)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    for _ in range(60):  # wall pieces and single stray cells
        r, c = rng.integers(2, 78), rng.integers(2, 118)
        occ[r:r + rng.integers(1, 8), c:c + rng.integers(1, 3)] = 100
    offsets = [(0.0, 0.0)] + [(LOS_POSE_TOLERANCE_M * 0.99 * math.cos(a), LOS_POSE_TOLERANCE_M * 0.99 * math.sin(a))
                              for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    for x, y, w in [(6.05, 4.05, 1.0), (3.35, 2.25, 0.3), (8.0, 5.5, 0.6)]:
        occ[int(y / 0.1) - 3:int(y / 0.1) + 4, int(x / 0.1) - 3:int(x / 0.1) + 4] = 0  # room for the object
        target = SimObject("t", "chair", x, y, width=w)
        world = World(GridMap(occ, 0.1, 0.0, 0.0), [target])
        vs = compute_viewshed(world.grid.blocking, (x, y), (0.0, 0.0), 0.1, 1.0, 3.5, obj_width=w)
        kept = camera = poses = missed = 0
        for row in range(vs.mask.shape[0]):
            for col in range(vs.mask.shape[1]):
                cx, cy = (vs.i0 + col + 0.5) * 0.1, (vs.j0 + row + 0.5) * 0.1
                d = math.hypot(cx - x, cy - y)
                if d < 1.0 + 1e-6 or d > 3.5 - 1e-6 or world.grid.blocking[vs.j0 + row, vs.i0 + col]:
                    continue  # (range edges: rounding)
                seen = world.line_of_sight(cx, cy, target)
                camera += seen
                if vs.mask[row, col]:
                    kept += 1
                    for dx, dy in offsets:
                        if not world.grid.blocked_cells(np.array([int((cy + dy) // 0.1)]), np.array([int((cx + dx) // 0.1)]))[0]:
                            poses += 1
                            missed += not world.line_of_sight(cx + dx, cy + dy, target)
        assert camera > 500 and kept >= 0.6 * camera, (x, y, w, kept, camera)  # (a cluttered test map)
        # Visibility is tested at cell centres, so a pose between them can still miss the object, but rarely
        assert missed <= 0.005 * poses, (x, y, w, missed, poses)
