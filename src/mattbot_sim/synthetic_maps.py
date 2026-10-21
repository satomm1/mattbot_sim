"""Synthetic simulation maps (clean walls, map_server layout), written by scripts/make_maps.py to mattbot_sim/maps/.

Each map is built from 2 m corridors, rooms beside them (0.2 m wall, a 1.2 m door) and alcoves (1.6 m niches opening
onto a corridor), so all three experiment obstacle types have places to go:
  blocking   a 1 m object closes a corridor to the planner (2 m - 1 m leaves less than the 0.8 m robot needs);
  on_route   a 0.5 m object against a corridor wall leaves room to pass;
  visible    alcoves and rooms off the corridors, in view of them.
Every corridor segment lies on a loop (blocking one leaves a way around); only room doors and the ladder's east wing
are dead ends. Coordinates are metres from the map origin (0, 0).

  grid_office  60 x 40 m: a 3 x 4 grid of corridors (six blocks), two rooms and an alcove per block. High
               redundancy: every blocked corridor has a short way around.
  ladder       84 x 34 m: two long corridors joined by four rungs, rooms on both sides and in the bays, a dead-end
               wing in the east. Low redundancy: a blocked rung means a long way round.
"""

import math

import numpy as np

RES = 0.05
CORRIDOR_M = 2.0
DOOR_M = 1.2
WALL_M = 0.2  # between a room and the corridor it opens onto
ALCOVE_W_M, ALCOVE_D_M = 1.6, 1.4


class Canvas:
    """Occupied (100) everywhere, with free space (0) carved out. Rows = y (ROS OccupancyGrid layout)."""

    def __init__(self, w_m, h_m):
        self.occupancy = np.full((int(round(h_m / RES)), int(round(w_m / RES))), 100, dtype=np.int16)

    def free(self, x0, y0, x1, y1):
        r0, r1 = int(round(min(y0, y1) / RES)), int(round(max(y0, y1) / RES))
        c0, c1 = int(round(min(x0, x1) / RES)), int(round(max(x0, x1) / RES))
        self.occupancy[max(r0, 0):r1, max(c0, 0):c1] = 0

    def hcorridor(self, y, x0, x1, w=CORRIDOR_M):
        """Corridor along y from x0 to x1 (centre line), w wide; it reaches w / 2 past both ends."""
        self.free(x0 - w / 2, y - w / 2, x1 + w / 2, y + w / 2)

    def vcorridor(self, x, y0, y1, w=CORRIDOR_M):
        self.free(x - w / 2, y0 - w / 2, x + w / 2, y1 + w / 2)

    def room(self, x0, y0, x1, y1, door_side, door_at, w=DOOR_M):
        """Room [x0, x1] x [y0, y1] with a door in its door_side wall (N, S, E, W) centred at door_at (x for N / S,
        y for E / W), cut through WALL_M (+ a little) so it opens onto the corridor next to that wall."""
        self.free(x0, y0, x1, y1)
        d = WALL_M + 0.1
        if door_side == "S":
            self.free(door_at - w / 2, y0 - d, door_at + w / 2, y0 + 0.01)
        elif door_side == "N":
            self.free(door_at - w / 2, y1 - 0.01, door_at + w / 2, y1 + d)
        elif door_side == "W":
            self.free(x0 - d, door_at - w / 2, x0 + 0.01, door_at + w / 2)
        elif door_side == "E":
            self.free(x1 - 0.01, door_at - w / 2, x1 + d, door_at + w / 2)
        else:
            raise ValueError(door_side)
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def alcove(self, x, y, side, w=ALCOVE_W_M, d=ALCOVE_D_M):
        """A niche w wide and d deep, opening at (x, y) on a corridor edge and going into the wall towards side."""
        if side == "N":
            self.free(x - w / 2, y - 0.01, x + w / 2, y + d)
        elif side == "S":
            self.free(x - w / 2, y - d, x + w / 2, y + 0.01)
        elif side == "E":
            self.free(x - 0.01, y - w / 2, x + d, y + w / 2)
        elif side == "W":
            self.free(x - d, y - w / 2, x + 0.01, y + w / 2)
        else:
            raise ValueError(side)


def grid_office():
    """(occupancy, landmarks {name: (x, y, theta)}, robot starts [(x, y, theta)])."""
    c = Canvas(60.0, 40.0)
    for y in (4.0, 20.0, 36.0):
        c.hcorridor(y, 4.0, 56.0)
    for x in (4.0, 20.0, 40.0, 56.0):
        c.vcorridor(x, 4.0, 36.0)
    # Corridor edges: y = 3/5, 19/21, 35/37; x = 3/5, 19/21, 39/41, 55/57. Rooms keep WALL_M from them.
    rooms = {
        "a_sw": c.room(5.2, 5.2, 11.0, 11.0, "S", 8.0),
        "a_ne": c.room(13.0, 13.0, 18.8, 18.8, "N", 16.0),
        "b_sw": c.room(21.2, 5.2, 29.0, 12.0, "W", 8.0),
        "b_ne": c.room(31.0, 12.0, 38.8, 18.8, "N", 35.0),
        "c_sw": c.room(41.2, 5.2, 48.0, 10.0, "S", 44.0),
        "c_e": c.room(49.0, 11.0, 54.8, 18.8, "E", 15.0),
        "d_nw": c.room(5.2, 28.0, 12.0, 34.8, "W", 31.0),
        "d_se": c.room(13.0, 21.2, 18.8, 27.0, "E", 24.0),
        "e_sw": c.room(21.2, 21.2, 28.0, 27.0, "S", 24.0),
        "e_ne": c.room(30.0, 28.0, 38.8, 34.8, "N", 34.0),
        "f_n": c.room(44.0, 27.0, 54.8, 34.8, "N", 49.0),
        "f_sw": c.room(41.2, 21.2, 47.0, 25.0, "W", 23.0),
    }
    c.alcove(5.0, 14.8, "E")   # off the west corridor into block A
    c.alcove(24.8, 19.0, "S")  # off the middle corridor into block B
    c.alcove(41.0, 14.8, "E")  # off x = 40 into block C
    c.alcove(14.8, 35.0, "S")  # off the north corridor into block D
    c.alcove(39.0, 23.8, "W")  # off x = 40 into block E
    c.alcove(50.8, 21.0, "N")  # off the middle corridor into block F
    landmarks = {name: (x, y, 0.0) for name, (x, y) in rooms.items()}
    landmarks["corner_ne"] = (56.0, 36.0, math.pi)
    landmarks["corner_sw"] = (4.0, 4.0, 0.0)
    starts = [(12.0, 4.0, 0.0), (30.0, 20.0, 0.0), (48.0, 36.0, math.pi), (4.0, 24.0, math.pi / 2),
              (56.0, 10.0, math.pi / 2)]
    return c.occupancy, landmarks, starts


def ladder():
    """(occupancy, landmarks, robot starts)."""
    c = Canvas(84.0, 34.0)
    c.hcorridor(6.0, 4.0, 76.0)
    c.hcorridor(24.0, 4.0, 76.0)
    for x in (4.0, 26.0, 50.0, 76.0):
        c.vcorridor(x, 6.0, 24.0, w=1.8)
    c.hcorridor(15.0, 76.0, 79.6)  # the east wing, a dead end
    # Corridor edges: y = 5/7, 23/25; rungs x = 3.1/4.9, 25.1/26.9, 49.1/50.9, 75.1/76.9; wing y = 14/16 to x = 80.6
    rooms = {
        "s_west": c.room(8.0, 0.8, 16.0, 4.8, "N", 12.0),
        "s_mid": c.room(34.0, 0.8, 42.0, 4.8, "N", 38.0),
        "s_east": c.room(58.0, 0.8, 66.0, 4.8, "N", 62.0),
        "n_west": c.room(10.0, 25.2, 18.0, 32.5, "S", 14.0),
        "n_mid": c.room(38.0, 25.2, 46.0, 32.5, "S", 42.0),
        "n_east": c.room(60.0, 25.2, 68.0, 32.5, "S", 64.0),
        "bay_w": c.room(8.0, 7.2, 15.0, 12.0, "S", 11.0),
        "bay_mid": c.room(30.0, 18.0, 38.0, 22.8, "N", 34.0),
        "bay_e": c.room(55.0, 7.2, 63.0, 12.0, "S", 59.0),
        "bay_ne": c.room(66.0, 16.0, 74.9, 22.8, "E", 19.0),
        "wing": c.room(80.8, 10.0, 83.5, 20.0, "W", 15.0),
    }
    c.alcove(20.8, 7.0, "N")   # off the south corridor into the west bay
    c.alcove(44.8, 23.0, "S")  # off the north corridor into the middle bay
    c.alcove(70.8, 5.0, "S")   # off the south corridor, south side
    c.alcove(49.1, 14.8, "W")  # off the x = 50 rung into the middle bay
    landmarks = {name: (x, y, 0.0) for name, (x, y) in rooms.items()}
    landmarks["corner_sw"] = (4.0, 6.0, 0.0)
    landmarks["corner_nw"] = (4.0, 24.0, 0.0)
    landmarks["corner_ne"] = (76.0, 24.0, math.pi)
    starts = [(18.0, 6.0, 0.0), (40.0, 24.0, math.pi), (54.0, 6.0, math.pi), (26.0, 15.0, math.pi / 2),
              (70.0, 24.0, 0.0)]
    return c.occupancy, landmarks, starts


MAPS = {"grid_office": grid_office, "ladder": ladder}


def write_map_server(occupancy, path_stem, resolution=RES, origin=(0.0, 0.0)):
    """<stem>.pgm / <stem>.yaml (and an identical <stem>_mod.pgm: no manual edits) in map_server's trinary format."""
    import os

    from PIL import Image

    img = np.full(occupancy.shape, 205, dtype=np.uint8)  # unknown
    img[occupancy == 0] = 254
    img[occupancy >= 50] = 0
    img = img[::-1]  # image row 0 is the top (max y)
    for suffix in ("", "_mod"):
        Image.fromarray(img).save(path_stem + suffix + ".pgm")
    with open(path_stem + ".yaml", "w") as f:
        f.write("image: %s.pgm\nresolution: %f\norigin: [%f, %f, 0.000000]\nnegate: 0\noccupied_thresh: 0.65\n"
                "free_thresh: 0.196\n" % (os.path.basename(path_stem), resolution, origin[0], origin[1]))
