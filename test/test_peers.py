"""Unit tests for mattbot_sim/peers.py (no ROS needed)."""

import json
import os
import sys

import pytest

from mattbot_sim.peers import peer_messages, peer_observations
from mattbot_sim.world import SimObject


def test_peer_observations_are_gapless_first_sightings():
    objs = [SimObject("a", "cone", 1.0, 2.0, 0.4), SimObject("b", "chair", 3.0, 4.0, 0.5)]
    obs = peer_observations(objs, 98, 1700000000, 1700000005.0)
    assert [o["seq"] for o in obs] == [1, 2]
    assert [o["obs_id"] for o in obs] == ["98-1700000000-1", "98-1700000000-2"]
    assert all(o["object_id"] == o["obs_id"] and o["observer_id"] == 98 for o in obs)
    assert (obs[1]["class_name"], obs[1]["x"], obs[1]["y"], obs[1]["width"]) == ("chair", 3.0, 4.0, 0.5)
    assert json.loads(peer_messages(objs, 98, 1, 2.0)[0])["stamp"] == 2.0


def test_peer_observation_matches_ledger_format():
    """The dicts must load as mattbot_dds dds_utils.ledger.Observation (skipped without mattbot_dds)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mattbot_dds", "src"))
    ledger = pytest.importorskip("dds_utils.ledger")
    (d,) = peer_observations([SimObject("a", "cone", 1.0, 2.0, 0.4)], 98, 5, 6.0)
    obs = ledger.Observation.from_dict(d)
    assert obs.obs_id == ledger.make_obs_id(98, 5, 1)
