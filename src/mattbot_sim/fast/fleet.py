"""One observation ledger shared by the whole fleet (instant, lossless communication).

Ego confirmations and removals follow mattbot_dds/scripts/observation_ledger.py (ego_callback,
ego_removal_callback / remove_object); peer reports follow remote_obs_callback. Beliefs follow
object_belief_map.py: objects_from_ledger over the ledger's canonical view (build_ledger_msg), then
object_state(hold, decay). Positions are in the one map frame (no transformation matrix in simulation).
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

from mattbot_sim.fast import imports  # noqa: F401
from dds_utils.belief import object_state, objects_from_ledger
from dds_utils.ledger import GatedResolver, Observation, ObservationLedger, Removal, make_obs_id


@dataclass
class Belief:
    """What /object_beliefs carries per object (the planner, evaluator and mapper read these)."""

    object_id: str
    class_name: str
    x: float
    y: float
    width: float
    belief: float


@dataclass
class FleetLedger:
    match_radius_m: float = 0.75
    hold_s: float = 20.0
    decay_s: float = 40.0
    session: int = 1
    ledger: ObservationLedger = field(default_factory=ObservationLedger)
    seq: Dict[int, int] = field(default_factory=dict)  # observer id -> last seq
    version: int = 0  # bumped on every change

    def __post_init__(self):
        self.resolver = GatedResolver(self.match_radius_m)
        self._beliefs = (None, None, [])  # (time, version, beliefs)

    def _next(self, robot_id):
        self.seq[robot_id] = self.seq.get(robot_id, 0) + 1
        return self.seq[robot_id]

    def confirm(self, robot_id, now, class_name, x, y, width, probability=0.0):
        """A robot confirmed an object (mapper confirmation, PRESENT check, re-sighting). Returns object_id."""
        seq = self._next(robot_id)
        obs_id = make_obs_id(robot_id, self.session, seq)
        obs = Observation(obs_id, obs_id, robot_id, self.session, seq, float(now), class_name, float(probability),
                          float(width), float(x), float(y))
        obs.object_id = self.resolver.resolve(obs, self.ledger)
        self.ledger.add(obs)
        self.version += 1
        return obs.object_id

    def remove_near(self, robot_id, now, class_name, x, y, width, probability=0.0):
        """A robot saw that an object is gone (ABSENT check): remove the nearest matching active object.
        Returns the removed object_id or None."""
        probe = Observation("", "", robot_id, self.session, 0, float(now), class_name, float(probability),
                            float(width), float(x), float(y))
        object_id = self.resolver.resolve(probe, self.ledger)
        members = self.ledger.objects().get(object_id) if object_id else None
        if not members:
            return None
        seq = self._next(robot_id)
        self.ledger.remove(Removal(make_obs_id(robot_id, self.session, seq), object_id, robot_id, self.session, seq,
                                   float(now), members[0].class_name, sum(o.x for o in members) / len(members),
                                   sum(o.y for o in members) / len(members)))
        self.version += 1
        return object_id

    def add_peer_observation(self, obs_dict):
        """An observation from outside the simulated fleet (scenario known_from_peer)."""
        obs = Observation.from_dict(obs_dict)
        if self.ledger.add(obs) and not self.ledger.absorb_stale(obs, self.resolver.radius_m):
            self.ledger.reconcile(obs, self.resolver)
        self.version += 1

    def beliefs(self, now) -> List[Belief]:
        """Active objects with their current belief (cached per (time, ledger version))."""
        t, version, cached = self._beliefs
        if t == now and version == self.version:
            return cached
        groups = self.ledger.query()
        canon = {o.object_id: self.ledger.canonical(o.object_id) for _agent, g in groups for o in g}
        entries = [SimpleNamespace(observations=[
            SimpleNamespace(object_id=canon[o.object_id], class_name=o.class_name, observer_id=o.observer_id,
                            x=o.x, y=o.y, local_x=o.x, local_y=o.y, width=o.width, stamp=o.stamp) for o in g])
            for _agent, g in groups]
        out = []
        for obj in objects_from_ledger(entries):
            s = object_state(obj, now, self.hold_s, self.decay_s)
            out.append(Belief(obj.object_id, obj.class_name, obj.x, obj.y, float(obj.width), float(s.belief)))
        out.sort(key=lambda b: b.belief)
        self._beliefs = (now, self.version, out)
        return out
