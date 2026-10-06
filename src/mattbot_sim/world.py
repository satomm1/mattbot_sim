"""Ground-truth 2D world: the static occupancy grid plus simple disc-shaped objects.

Pure Python + numpy (no ROS) so it can be unit-tested.

Map conventions follow nav_msgs/OccupancyGrid: row 0 is at origin_y, values -1 unknown, 0 free, 100 occupied.
Unknown cells block rays (the same as observation_evaluator's background raycast).
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml


@dataclass
class GridMap:
    occupancy: np.ndarray  # HxW int16
    resolution: float  # m per cell
    origin_x: float
    origin_y: float
    blocking: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.occupancy = np.asarray(self.occupancy, dtype=np.int16)
        self.blocking = (self.occupancy >= 50) | (self.occupancy < 0)

    @property
    def height(self):
        return self.occupancy.shape[0]

    @property
    def width(self):
        return self.occupancy.shape[1]

    def to_cell(self, x, y):
        """World (m) -> (row, col); may be out of bounds."""
        return int(math.floor((y - self.origin_y) / self.resolution)), int(math.floor((x - self.origin_x) / self.resolution))

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x, y):
        row, col = self.to_cell(x, y)
        return self.in_bounds(row, col) and self.occupancy[row, col] == 0

    def blocked_cells(self, rows, cols):
        """Vectorised: True where (rows, cols) is blocking or outside the map."""
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        out = np.ones(rows.shape, dtype=bool)
        out[inside] = self.blocking[rows[inside], cols[inside]]
        return out


def load_json_map(path):
    """mattbot_mcl/map_json/*.json, the format entry_exit.py publishes on the robots."""
    with open(path, "r") as f:
        m = json.load(f).get("data", {}).get("map", {})
    occ = np.asarray(m["occupancy"], dtype=np.int16).reshape(m["height"], m["width"])
    return GridMap(occ, float(m["resolution"]), float(m.get("origin_x", 0.0)), float(m.get("origin_y", 0.0)))


def load_yaml_map(path, image=None):
    """map_server YAML + PGM (trinary interpretation). ``image`` overrides the YAML's image file."""
    from PIL import Image

    with open(path, "r") as f:
        meta = yaml.safe_load(f)
    image_path = image or meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(path), image_path)
    pixels = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    p = pixels / 255.0 if meta.get("negate", 0) else (255.0 - pixels) / 255.0
    occ = np.full(pixels.shape, -1, dtype=np.int16)
    occ[p > float(meta.get("occupied_thresh", 0.65))] = 100
    occ[p < float(meta.get("free_thresh", 0.196))] = 0
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    return GridMap(occ[::-1].copy(), float(meta["resolution"]), float(origin[0]), float(origin[1]))


def load_maps(spec, map_json_dir):
    """Load (map, map_mod) as the robots do.

    ``spec`` is either a JSON map stem in ``map_json_dir`` (``current_map`` -> current_map.json and
    current_map_mod.json), or a path to a map_server .yaml (its ``<image stem>_mod.pgm`` is used for map_mod).
    The mod map falls back to the plain map when no variant exists.
    """
    if spec.endswith(".yaml"):
        grid = load_yaml_map(spec)
        with open(spec, "r") as f:
            image = yaml.safe_load(f)["image"]
        stem, ext = os.path.splitext(image)
        mod_image = os.path.join(os.path.dirname(spec), stem + "_mod" + ext)
        grid_mod = load_yaml_map(spec, mod_image) if os.path.exists(mod_image) else grid
        return grid, grid_mod
    stem = spec[:-5] if spec.endswith(".json") else spec
    path = stem if os.path.isabs(stem) else os.path.join(map_json_dir, stem)
    grid = load_json_map(path + ".json")
    grid_mod = load_json_map(path + "_mod.json") if os.path.exists(path + "_mod.json") else grid
    return grid, grid_mod


@dataclass
class SimObject:
    object_id: str
    class_name: str
    x: float
    y: float
    width: float = 0.5  # footprint diameter (m)
    present: bool = True

    @property
    def radius(self):
        return 0.5 * self.width


@dataclass
class Placement:
    """Ground-truth interval during which an object stood at one spot (for scoring)."""

    object_id: str
    class_name: str
    x: float
    y: float
    width: float
    t_start: float
    t_end: Optional[float] = None  # None while still there

    def to_dict(self):
        return dict(self.__dict__)


class World:
    def __init__(self, grid, objects=(), now=0.0):
        self.grid = grid
        self.objects: Dict[str, SimObject] = {}
        self.placements: List[Placement] = []
        for obj in objects:
            self._add(obj, now)

    # ---------- Objects ----------

    def present_objects(self):
        return [o for o in self.objects.values() if o.present]

    def _add(self, obj, now):
        if obj.object_id in self.objects:
            raise ValueError("duplicate object id %r" % obj.object_id)
        self.objects[obj.object_id] = obj
        if obj.present:
            self._open(obj, now)

    def _open(self, obj, now):
        self.placements.append(Placement(obj.object_id, obj.class_name, obj.x, obj.y, obj.width, now))

    def _close(self, object_id, now):
        for p in self.placements:
            if p.object_id == object_id and p.t_end is None:
                p.t_end = now

    def _get(self, object_id):
        if object_id not in self.objects:
            raise ValueError("unknown object %r (known: %s)" % (object_id, ", ".join(sorted(self.objects))))
        return self.objects[object_id]

    def apply_event(self, event, now):
        """Apply one world event (see scenario.parse_event). Returns a human-readable description."""
        if "remove" in event:
            obj = self._get(event["remove"])
            if obj.present:
                obj.present = False
                self._close(obj.object_id, now)
            return "removed %s (%s) from (%.2f, %.2f)" % (obj.object_id, obj.class_name, obj.x, obj.y)
        if "restore" in event:
            obj = self._get(event["restore"])
            if not obj.present:
                obj.present = True
                self._open(obj, now)
            return "restored %s (%s) at (%.2f, %.2f)" % (obj.object_id, obj.class_name, obj.x, obj.y)
        if "move" in event:
            obj = self._get(event["move"])
            old = (obj.x, obj.y)
            self._close(obj.object_id, now)
            obj.x, obj.y, obj.present = float(event["x"]), float(event["y"]), True
            self._open(obj, now)
            return "moved %s (%s) from (%.2f, %.2f) to (%.2f, %.2f)" % (obj.object_id, obj.class_name, old[0], old[1], obj.x, obj.y)
        if "add" in event:
            obj = event["add"]
            self._add(obj, now)
            return "added %s (%s) at (%.2f, %.2f)" % (obj.object_id, obj.class_name, obj.x, obj.y)
        raise ValueError("unsupported event %r" % (event,))

    # ---------- Geometry ----------

    def raycast(self, x, y, angles, max_range, include_objects=True, exclude=()):
        """Distance along each ray to the first blocking map cell or present object; np.inf if none within max_range."""
        angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        step = 0.5 * self.grid.resolution
        d = np.arange(step, max_range + step, step)
        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        rows = np.floor((y + d[None, :] * sin - self.grid.origin_y) / self.grid.resolution).astype(np.int64)
        cols = np.floor((x + d[None, :] * cos - self.grid.origin_x) / self.grid.resolution).astype(np.int64)
        hit = self.grid.blocked_cells(rows, cols)
        first = np.argmax(hit, axis=1)
        ranges = np.where(hit[np.arange(len(angles)), first], d[first] - step, np.inf)
        if include_objects:
            for obj in self.present_objects():
                if obj.object_id in exclude:
                    continue
                ranges = np.minimum(ranges, ray_circle(x, y, angles, obj.x, obj.y, obj.radius))
        ranges[ranges > max_range] = np.inf
        return ranges

    def line_of_sight(self, x0, y0, target, margin=0.05):
        """True if nothing blocks the view from (x0, y0) to the near side of ``target`` (a SimObject)."""
        dist = math.hypot(target.x - x0, target.y - y0)
        reach = dist - target.radius - margin  # the ray may end at the object's surface
        if reach <= 0.0:
            return True
        angle = math.atan2(target.y - y0, target.x - x0)
        hit = self.raycast(x0, y0, [angle], reach, exclude=(target.object_id,))[0]
        return not np.isfinite(hit)

    def collides(self, x, y, radius):
        """True if a disc of ``radius`` at (x, y) overlaps a blocking cell or a present object."""
        res = self.grid.resolution
        n = int(math.ceil(radius / res))
        offs = np.arange(-n, n + 1) * res
        dx, dy = np.meshgrid(offs, offs)
        inside = dx ** 2 + dy ** 2 <= radius ** 2
        rows = np.floor((y + dy[inside] - self.grid.origin_y) / res).astype(np.int64)
        cols = np.floor((x + dx[inside] - self.grid.origin_x) / res).astype(np.int64)
        if self.grid.blocked_cells(rows, cols).any():
            return True
        return any(math.hypot(o.x - x, o.y - y) < o.radius + radius for o in self.present_objects())


def ray_circle(x, y, angles, cx, cy, r):
    """Distance along rays from (x, y) to a circle; np.inf on a miss, 0 if the origin is inside it."""
    angles = np.asarray(angles, dtype=np.float64)
    ux, uy = np.cos(angles), np.sin(angles)
    vx, vy = cx - x, cy - y
    t = vx * ux + vy * uy  # projection of the centre onto the ray
    perp2 = vx * vx + vy * vy - t * t
    disc = r * r - perp2
    out = np.full(angles.shape, np.inf)
    if vx * vx + vy * vy <= r * r:
        return np.zeros(angles.shape)
    hit = (disc >= 0.0) & (t > 0.0)
    out[hit] = t[hit] - np.sqrt(disc[hit])
    return out
