"""Simulated peer reports: scenario objects the ledger learns from another robot, not from this one's camera.

Detour targets are by design not visible from the patrol path, so the robot would never add them to its
ledger itself. scenario_runner publishes them on /ledger/observation_from_agent as if a peer robot had
seen them, which is how the fleet learns about such objects. Each message is the JSON of a
mattbot_dds dds_utils.ledger.Observation (same fields; built here so it can be tested without mattbot_dds).
Positions are taken as the reference frame; the sim has no /transformation_matrix, so the ledger uses
them unchanged as local map positions. Pure Python (no ROS).
"""

import json


def obs_id(observer_id, session, seq):
    """Same format as dds_utils.ledger.make_obs_id."""
    return "%d-%d-%d" % (observer_id, session, seq)


def peer_observations(objects, peer_id, session, stamp):
    """Observation dicts for ``objects`` (SimObjects), seq 1..n with no gaps (the ledger asks for a resync
    when a peer's seqs have holes). The object id is the obs id, as for a peer's first sighting."""
    out = []
    for seq, o in enumerate(objects, start=1):
        oid = obs_id(peer_id, session, seq)
        out.append({
            "obs_id": oid,
            "object_id": oid,
            "observer_id": int(peer_id),
            "session": int(session),
            "seq": seq,
            "stamp": float(stamp),
            "class_name": o.class_name,
            "probability": 1.0,
            "width": float(o.width),
            "x": float(o.x),
            "y": float(o.y),
        })
    return out


def peer_messages(objects, peer_id, session, stamp):
    """JSON strings for /ledger/observation_from_agent."""
    return [json.dumps(d) for d in peer_observations(objects, peer_id, session, stamp)]
