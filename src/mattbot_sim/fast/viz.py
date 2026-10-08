"""Pictures of a fast-sim run: an animated GIF of snapshots and one overview PNG (matplotlib + Pillow).

sim.frames = [] before sim.run() makes the simulator append snapshot(sim) every sim.frame_every seconds.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:red"]
OUTCOME_MARKERS = {"PRESENT": ("o", "green"), "ABSENT": ("X", "red"), "INCONCLUSIVE": ("s", "grey")}


def snapshot(sim):
    robots = []
    for rid, r in sim.robots.items():
        target = r.nav.observe_target[1] if r.nav.observe_target else None
        robots.append((rid, r.pose.x, r.pose.y, r.pose.theta, r.nav.mode, target))
    return {"t": sim.now, "robots": robots,
            "objects": [(o.x, o.y, o.present) for o in sim.world.objects.values()],
            "beliefs": [(b.x, b.y, b.belief) for b in sim.beliefs()]}


def crop(sim, margin=3.0):
    xs, ys = [], []
    for r in sim.robots.values():
        xs += [r.scenario.start.x] + [w.x for w in r.scenario.waypoints]
        ys += [r.scenario.start.y] + [w.y for w in r.scenario.waypoints]
    xs += [o.x for o in sim.world.objects.values()]
    ys += [o.y for o in sim.world.objects.values()]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _axes(sim, box, width_in=10.0):
    x0, x1, y0, y1 = box
    fig, ax = plt.subplots(figsize=(width_in, max(width_in * (y1 - y0) / (x1 - x0), 2.5) + 0.6))
    g = sim.grid
    # Only the shown part of the map (drawing the whole map in every GIF frame is slow)
    c0, c1 = max(int((x0 - g.origin_x) / g.resolution), 0), min(int((x1 - g.origin_x) / g.resolution) + 1, g.width)
    r0, r1 = max(int((y0 - g.origin_y) / g.resolution), 0), min(int((y1 - g.origin_y) / g.resolution) + 1, g.height)
    occ = g.occupancy[r0:r1, c0:c1]
    shade = (occ >= 50) * 1.0 + (occ < 0) * 0.35  # walls dark, unknown light
    extent = (g.origin_x + c0 * g.resolution, g.origin_x + c1 * g.resolution,
              g.origin_y + r0 * g.resolution, g.origin_y + r1 * g.resolution)
    ax.imshow(shade, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1.2, interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return fig, ax


def render_gif(sim, frames, path, fps=10, dpi=80):
    fig, ax = _axes(sim, crop(sim))
    color = {rid: COLORS[k % len(COLORS)] for k, rid in enumerate(sim.robots)}
    dynamic = []

    def draw(frame):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()
        for x, y, present in frame["objects"]:
            dynamic.append(ax.scatter([x], [y], s=60, marker="s", c="green" if present else "lightgrey",
                                      edgecolors="k", zorder=3))
        for x, y, b in frame["beliefs"]:
            dynamic.append(ax.add_patch(plt.Circle((x, y), 0.45, fill=False, lw=2, color=plt.cm.autumn_r(b),
                                                   zorder=4)))
        for rid, x, y, th, mode, target in frame["robots"]:
            c = color[rid]
            dynamic.append(ax.add_patch(plt.Circle((x, y), 0.2, color=c, zorder=5)))
            dynamic.extend(ax.plot([x, x + 0.5 * math.cos(th)], [y, y + 0.5 * math.sin(th)], color="k", lw=1.5, zorder=6))
            if target is not None:
                dynamic.extend(ax.plot([x, target[0]], [y, target[1]], color=c, ls="--", lw=1, zorder=4))
            dynamic.append(ax.text(x, y + 0.4, "%d %s" % (rid, mode.lower()), fontsize=7, ha="center", zorder=7))
        dynamic.append(ax.text(0.01, 0.98, "t = %.0f s" % frame["t"], transform=ax.transAxes, va="top",
                               fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.8), zorder=8))

    ax.set_title("%s   (squares: true objects, rings: ledger belief red=1 -> yellow=0)" % sim.scenario.name,
                 fontsize=9)
    images = []
    fig.set_dpi(dpi)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)  # map and axes, drawn once
    for frame in frames:  # blit: restore the background, draw only this frame's markers
        draw(frame)
        fig.canvas.restore_region(background)
        for artist in dynamic:
            ax.draw_artist(artist)
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=128))
    plt.close(fig)
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def render_png(sim, frames, path):
    fig, ax = _axes(sim, crop(sim))
    for k, (rid, r) in enumerate(sim.robots.items()):
        c = COLORS[k % len(COLORS)]
        track = [(x, y) for f in frames for (i, x, y, *_rest) in f["robots"] if i == rid]
        if track:
            ax.plot([p[0] for p in track], [p[1] for p in track], color=c, lw=1, alpha=0.7, label="robot %d" % rid)
        for res, stop in zip(r.scorer.results, [s for s in r.scorer.stop_log if s["event"] == "ENDED"]):
            marker, mc = OUTCOME_MARKERS.get(res["outcome"], ("o", "k"))
            if stop.get("robot_xy"):
                ax.scatter(*stop["robot_xy"], marker=marker, c=mc, edgecolors=c, s=50, zorder=5)
    for o in sim.world.objects.values():
        ax.scatter([o.x], [o.y], s=70, marker="s", c="green" if o.present else "lightgrey", edgecolors="k", zorder=4)
        ax.annotate(o.object_id, (o.x, o.y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("%s: tracks and checks (o present, X absent, s inconclusive, at the robot's position); "
                 "grey square = removed" % sim.scenario.name, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
