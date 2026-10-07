"""The shipped detour scenarios really test what they claim (offline, mattbot_sim/detour_check.py).

Objects annotated ``detour: expected`` must be hidden from the patrol path (and from the sim camera)
and worth a detour with the sim.launch defaults; ``detour: declined`` ones must not get one. Needs
mattbot_navigation's navigation_utils and path_planning's social_path_planning; skipped otherwise.
Takes a few seconds (roadmap build into a temporary cache).
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub in ("mattbot_navigation/src", "path_planning/src"):
    sys.path.insert(0, os.path.join(SRC, sub))
pytest.importorskip("navigation_utils.detour")

from mattbot_sim.detour_check import check_scenario  # noqa: E402
from mattbot_sim.scenario import load_scenario  # noqa: E402

SCENARIOS = os.path.join(HERE, "..", "scenarios")
MAP_DIR = os.path.join(SRC, "mattbot_mcl", "map_json")
ANNOTATED = sorted(
    f for f in os.listdir(SCENARIOS)
    if f.endswith(".yaml") and load_scenario(os.path.join(SCENARIOS, f)).detour
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("roadmap"))


@pytest.mark.skipif(not os.path.isdir(MAP_DIR), reason="mattbot_mcl/map_json not found")
@pytest.mark.parametrize("name", ANNOTATED)
def test_detour_annotations(name, cache_dir):
    sc = load_scenario(os.path.join(SCENARIOS, name))
    results = {r.object_id: r for r in check_scenario(sc, MAP_DIR, cache_dir, object_ids=set(sc.detour))}
    for oid, want in sc.detour.items():
        r = results[oid]
        assert r.hidden, "%s is visible from the patrol path" % oid
        assert not r.camera_sees_from_path, "the sim camera can see %s from the path" % oid
        assert oid in sc.known_from_peer, "%s is hidden, so only a peer report can put it in the ledger" % oid
        if want == "expected":
            assert r.detour_go, "the planner would not detour to %s" % oid
        else:
            assert not r.detour_go, "the planner would detour to %s" % oid
