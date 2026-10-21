#!/usr/bin/env python3
"""Write the synthetic simulation maps (src/mattbot_sim/synthetic_maps.py) and their landmark files.

  make_maps.py            # mattbot_sim/maps/<name>.{yaml,pgm} + <name>_mod.pgm, experiments/landmarks_<name>.yaml
  make_maps.py --png DIR  # also a review PNG per map (walls, landmarks, robot starts)

The maps are deterministic: re-running rewrites the same files. Scenarios and experiment specs use them by name
(map: grid_office); mattbot_sim.world.load_maps finds them in mattbot_sim/maps/.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
PKG = os.path.normpath(os.path.join(HERE, ".."))


def main():
    from mattbot_sim.synthetic_maps import MAPS, write_map_server

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", default="", help="directory for review PNGs")
    args = ap.parse_args()
    os.makedirs(os.path.join(PKG, "maps"), exist_ok=True)
    for name, build in MAPS.items():
        occupancy, landmarks, starts = build()
        write_map_server(occupancy, os.path.join(PKG, "maps", name))
        lm_path = os.path.join(PKG, "experiments", "landmarks_%s.yaml" % name)
        with open(lm_path, "w") as f:
            f.write("# Task destinations on the synthetic map %s (mattbot_sim/maps, scripts/make_maps.py): [x, y, theta]\n"
                    % name)
            for lm, (x, y, th) in landmarks.items():
                f.write("%s: [%.2f, %.2f, %.4f]\n" % (lm, x, y, th))
        print("%s: %.0f x %.0f m, %d landmarks -> maps/%s.yaml, experiments/landmarks_%s.yaml" % (
            name, occupancy.shape[1] * 0.05, occupancy.shape[0] * 0.05, len(landmarks), name, name))
        print("  robot starts: %s" % ", ".join("(%.1f, %.1f, %.2f)" % s for s in starts))
        if args.png:
            _png(name, occupancy, landmarks, starts, os.path.join(args.png, name + ".png"))


def _png(name, occupancy, landmarks, starts, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    h, w = occupancy.shape
    fig, ax = plt.subplots(figsize=(w * 0.05 / 4 + 2, h * 0.05 / 4 + 1))
    ax.imshow(np.where(occupancy == 0, 1.0, 0.0), cmap="gray", origin="lower", extent=(0, w * 0.05, 0, h * 0.05))
    for lm, (x, y, _th) in landmarks.items():
        ax.plot(x, y, "b^")
        ax.annotate(lm, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7, color="b")
    for k, (x, y, _th) in enumerate(starts):
        ax.plot(x, y, "go")
        ax.annotate("start %d" % k, (x, y), textcoords="offset points", xytext=(3, -9), fontsize=7, color="g")
    ax.set_title(name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
