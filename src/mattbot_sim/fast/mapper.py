"""Per-robot occupancy_grid_mapper rules (mattbot_navigation/scripts/occupancy_grid_mapper.py):
detections -> proposals -> confirmed objects with a blockout in the robot's object layer.

- No processing in unstable modes (LOCALIZING, LOCALIZING2, PARK_HEADING, BACKING, WAITING_FOR_INIT)
  and none for classes "unknown" / "person".
- A detection within one width of a confirmed object is ignored. Otherwise it matches a proposal of the same
  class within width / 2 of the proposal's first position; seen more than 5 times -> confirmed. A match that
  does not confirm still appends a new proposal (as the mapper does). Proposals not matched in more than 10
  frames are dropped.
- Confirmation: square blockout of side min(width, 0.5) and one confirmation to the ledger.
- Blockouts expire 30 s after they were added, checked only while the robot is IDLE (the mapper's expiry
  timer); a later sighting then confirms the object again, which refreshes its belief.
- Objects confirmed by other robots are added at once (data_subscriber -> /object_from_agent).
- With ledger_blockout, ledger objects are blocked as well (object_layers.ledger_blockout_grid).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401
from navigation_utils.object_layers import ledger_blockout_grid

UNSTABLE_MODES = ("LOCALIZING", "LOCALIZING2", "PARK_HEADING", "BACKING", "WAITING_FOR_INIT")
IGNORED_CLASSES = ("unknown", "person")


@dataclass
class Confirmed:
    x: float
    y: float
    width: float  # blockout side
    t_added: float


@dataclass
class Mapper:
    shape: tuple  # (H, W) of the map
    origin: tuple
    resolution: float
    confirm_count: int = 6
    miss_limit: int = 10
    ttl_s: float = 30.0
    max_blockout_m: float = 0.5
    confirmed: List[Confirmed] = field(default_factory=list)
    proposals: list = field(default_factory=list)  # [x, y, width, seen, missed, matched_this_frame, class]

    def __post_init__(self):
        self.detected_layer = np.zeros(self.shape, dtype=np.int16)
        self.ledger_layer = None
        self.version = 0
        self._layer = None

    # ---------- Detections ----------

    def frame(self, now, mode_name, detections):
        """One detector frame. Returns [(class, x, y, width)] newly confirmed (to send to the ledger)."""
        if mode_name in UNSTABLE_MODES:
            return []
        new = []
        for d in detections:
            if d.class_name in IGNORED_CLASSES:
                continue
            if any(np.hypot(c.x - d.x, c.y - d.y) < d.width for c in self.confirmed):
                continue
            match_proposed = False
            for ii in range(len(self.proposals) - 1, -1, -1):
                p = self.proposals[ii]
                if p[6] != d.class_name or np.hypot(p[0] - d.x, p[1] - d.y) >= d.width / 2:
                    continue
                p[3] += 1
                p[5] = True
                if p[3] >= self.confirm_count:
                    match_proposed = True
                    self.proposals.pop(ii)
                    self.add_confirmed(now, d.x, d.y, d.width)
                    new.append((d.class_name, d.x, d.y, d.width))
                    break
            if not match_proposed:
                self.proposals.append([d.x, d.y, d.width, 0, 0, True, d.class_name])
        for ii in range(len(self.proposals) - 1, -1, -1):
            p = self.proposals[ii]
            if not p[5]:
                p[4] += 1
                if p[4] > self.miss_limit:
                    self.proposals.pop(ii)
            else:
                p[5] = False
        return new

    def add_confirmed(self, now, x, y, width):
        w = min(width, self.max_blockout_m)
        self.confirmed.append(Confirmed(x, y, w, now))
        self._paint(x, y, w, 100)

    def add_from_agent(self, now, x, y, width):
        """Another robot confirmed an object (/object_from_agent): no confirmation step."""
        if any(np.hypot(c.x - x, c.y - y) < width for c in self.confirmed):
            return
        self.add_confirmed(now, x, y, width)

    def expire(self, now, mode_name):
        """The mapper's 1 Hz expiry timer: only while IDLE."""
        if mode_name != "IDLE":
            return
        keep = [c for c in self.confirmed if now - c.t_added <= self.ttl_s]
        if len(keep) != len(self.confirmed):
            self.confirmed = keep
            self.detected_layer[:] = 0
            for c in keep:
                self._paint(c.x, c.y, c.width, 100)

    def _paint(self, x, y, w, value):
        ox, oy = self.origin
        i0, i1 = int((x - ox - w / 2) / self.resolution), int((x - ox + w / 2) / self.resolution)
        j0, j1 = int((y - oy - w / 2) / self.resolution), int((y - oy + w / 2) / self.resolution)
        h, wd = self.shape
        self.detected_layer[max(j0, 0):min(j1, h), max(i0, 0):min(i1, wd)] = value
        self.version += 1
        self._layer = None

    # ---------- Layers ----------

    def set_ledger_objects(self, beliefs):
        """ledger_blockout: block every ledger object (belief >= 0)."""
        h, w = self.shape
        grid = ledger_blockout_grid([(b.x, b.y, b.width, b.belief) for b in beliefs], w, h, self.resolution,
                                    self.origin)
        if self.ledger_layer is None or not np.array_equal(grid, self.ledger_layer):
            self.ledger_layer = grid
            self.version += 1
            self._layer = None

    def object_layer(self):
        """/object_map: detected blockouts, max with the ledger layer (HxW, 100 = blocked)."""
        if self._layer is None:
            layer = self.detected_layer
            if self.ledger_layer is not None:
                layer = np.maximum(layer, self.ledger_layer.astype(np.int16))
            self._layer = layer
        return self._layer

    def is_blocked(self, x, y):
        col, row = int((x - self.origin[0]) / self.resolution), int((y - self.origin[1]) / self.resolution)
        h, w = self.shape
        return 0 <= row < h and 0 <= col < w and self.object_layer()[row, col] >= 50
