"""The synthetic simulation maps (src/mattbot_sim/synthetic_maps.py, maps/*.yaml): they load by name, match their
generator, keep every landmark reachable, and give the experiment generator places for every obstacle type."""

import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

pytest.importorskip("scipy")
from mattbot_sim.fast import imports  # noqa: E402,F401
from mattbot_sim.synthetic_maps import MAPS  # noqa: E402

NAMES = sorted(MAPS)


@pytest.mark.parametrize("name", NAMES)
def test_map_files_match_the_generator(name):
    """maps/<name>.pgm is what make_maps.py writes now (re-run it after changing synthetic_maps.py)."""
    from mattbot_sim.world import load_maps

    grid, grid_mod = load_maps(name, imports.MAP_JSON_DIR)  # by name: mattbot_sim/maps/<name>.yaml
    occupancy, _landmarks, _starts = MAPS[name]()
    assert np.array_equal(grid.occupancy, occupancy) and np.array_equal(grid_mod.occupancy, occupancy)
    assert (grid.origin_x, grid.origin_y, grid.resolution) == (0.0, 0.0, 0.05)


@pytest.mark.parametrize("name", NAMES)
def test_free_space_is_connected_and_landmarks_are_reachable(name):
    from scipy import ndimage

    from mattbot_sim.experiment.generate import MapContext, snap_landmarks, unreachable_landmarks
    from mattbot_sim.experiment.spec import load_landmarks

    occupancy, _landmarks, starts = MAPS[name]()
    _labels, n = ndimage.label(occupancy == 0)
    assert n == 1  # one free region: every room and alcove opens onto a corridor
    ctx = MapContext(name)
    landmarks = load_landmarks(os.path.join(PKG, "experiments", "landmarks_%s.yaml" % name))
    assert snap_landmarks(ctx, landmarks) == landmarks  # every landmark is already free for the planner
    for x, y, _th in starts:
        assert ctx.is_free(x, y)
        assert unreachable_landmarks(ctx, landmarks, (x, y)) == []


@pytest.mark.parametrize("name", NAMES)
def test_every_obstacle_type_has_places(name):
    from mattbot_sim.experiment.generate import make_instance, setup
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(os.path.join(PKG, "experiments", "rechecking", "maps", name + ".yaml"))
    assert spec.map == name and len(spec.robots) == 3
    _ctx, landmarks, cands, conn = setup(spec)
    for typ, cfg in spec.obstacle_types.items():
        top = cfg["max_importance_m"]
        usable = [c for c in cands if c["type"] == typ and c["I_o"] >= cfg["min_importance_m"]
                  and (top is None or c["I_o"] <= top)]
        assert len(usable) >= 20, (typ, len(usable))
    inst = make_instance(spec, 0, cands, landmarks, conn)
    assert [ob["type"] for ob in inst["obstacles"]] == ["blocking"] * 2 + ["on_route"] * 2 + ["visible"] * 2
