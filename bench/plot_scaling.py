"""plot_scaling.py: one figure from bench_scale_full.py's JSON.

Eight panels, each with a single y-axis:

  A,B  recall vs batch throughput (Pareto frontier), one curve per scale
  C    build time vs N
  D    index memory vs N (should be linear)
  E    vectors scanned at fixed recall vs N
  F    nprobe at fixed recall vs N
  G    p50 latency at fixed recall vs N
  H    tree depth vs N

N uses a sequential single-hue ramp; the two profiles use blue #2a78d6 and orange #eb6834, which
stay distinguishable for colour-blind readers.

    python bench/plot_scaling.py --json bench/results/scaling_full.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.cm import ScalarMappable                       # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
SERIES = {"memory": "#2a78d6", "quality": "#eb6834"}
# Use marker shape as well as colour: where the two profiles overlap (routing is shared, so panels
# E/F often do) one would otherwise hide the other.
DASH = {"memory": "-", "quality": "--"}
MARK = {"memory": "o", "quality": "s"}
# sequential ramps, one hue each, light -> dark
RAMPS = {
    "memory": LinearSegmentedColormap.from_list("m", ["#bcd6f5", "#12385f"]),
    "quality": LinearSegmentedColormap.from_list("q", ["#f7c4ad", "#7a2c0f"]),
}


def interp_at(rows, target, field):
    pts = sorted((r["recall@10"], r[field]) for r in rows)
    for i in range(1, len(pts)):
        (r0, v0), (r1, v1) = pts[i - 1], pts[i]
        if r0 <= target <= r1 and r1 > r0:
            f = (target - r0) / (r1 - r0)
            return v0 + f * (v1 - v0)
    return None


def alpha(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    return float(np.polyfit(lx, ly, 1)[0])


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="600", pad=8,
                 loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)
    ax.grid(True, color="#e6e5e0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cd")
        ax.spines[s].set_linewidth(0.8)


def fmtN(v):
    if v >= 1e6:
        return f"{round(v / 1e6, 1):g}M"
    if v >= 1e3:
        return f"{round(v / 1e3):g}k"
    return str(int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="bench/results/scaling_full.json")
    ap.add_argument("--out", default="bench/results/scaling.png")
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args()

    data = json.load(open(a.json))
    runs = [r for r in data["runs"] if "sweep" in r]
    profs = [p for p in ("memory", "quality")
             if any(r["profile"] == p for r in runs)]
    by = {p: sorted([r for r in runs if r["profile"] == p],
                    key=lambda r: r["n"]) for p in profs}

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.045, right=.988, top=.845, bottom=.085,
                        wspace=.30, hspace=.42)

    # --- A, B: recall vs throughput, one curve per scale ---
    for i, p in enumerate(profs[:2]):
        ax = axes[0][i]
        rs = by[p]
        lo, hi = np.log10(rs[0]["n"]), np.log10(rs[-1]["n"])
        for r in rs:
            t = 0.12 + 0.88 * ((np.log10(r["n"]) - lo) / max(hi - lo, 1e-9))
            c = RAMPS[p](t)
            xs = [w["batch_qps"] for w in r["sweep"]]
            ys = [w["recall@10"] for w in r["sweep"]]
            ax.plot(xs, ys, "-o", color=c, lw=2.0, ms=3.6, zorder=3)
        # N gets a colour ramp with a colourbar instead of inline labels; every curve ends in the
        # same corner of a Pareto plot, so labels collide.
        ns = [r["n"] for r in rs]
        sm = ScalarMappable(norm=LogNorm(vmin=min(ns), vmax=max(ns)),
                            cmap=RAMPS[p])
        cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cb.set_ticks(ns)
        cb.set_ticklabels([fmtN(v) for v in ns])
        cb.ax.tick_params(labelsize=7.2, colors=INK2, length=2, width=0.6)
        cb.outline.set_visible(False)
        ax.set_xscale("log")
        ax.set_ylim(min(0.75, ax.get_ylim()[0]), 1.005)
        style(ax, "batch throughput (qps, log)", "recall@10",
              f"{'AB'[i]}. Pareto frontier - {p} profile")

    # --- per-scale derived series ---
    panels = [
        (axes[0][2], "build", "C. Build time vs N", "seconds",
         lambda r: r["build"]["total_s"], True),
        (axes[0][3], "mem", "D. Index memory vs N", "MB",
         lambda r: r["index_mb"], True),
        (axes[1][0], "targets", "E. nprobe vs N at several recall targets",
         "cells probed", None, True),
        (axes[1][1], "np", f"F. nprobe @ R@10>={a.target}", "cells probed",
         None, True),
        (axes[1][2], "lat", f"G. p50 latency @ R@10>={a.target}", "ms",
         None, True),
        (axes[1][3], "depth", "H. Centroid-tree depth vs N", "levels",
         lambda r: r["tree_depth"], False),
    ]
    field = {"scan": "scanned_est", "np": "nprobe", "lat": "p50_ms"}

    for ax, kind, title, ylab, getter, logy in panels:
        notes = []
        if kind == "targets":
            # The scaling exponent depends on the recall target, so each target gets its own line,
            # coloured on a sequential ramp.
            rs = by["memory"]
            tgts = [0.70, 0.80, 0.90, 0.95]
            for ti, tg in enumerate(tgts):
                xs, ys = [], []
                for r in rs:
                    v = interp_at(r["sweep"], tg, "nprobe")
                    if v is None:
                        continue
                    xs.append(r["n"])
                    ys.append(v)
                if len(xs) < 2:
                    continue
                c = RAMPS["memory"](0.20 + 0.80 * ti / max(len(tgts) - 1, 1))
                ax.plot(xs, ys, "-o", color=c, lw=2.0, ms=4.5, zorder=3)
                ax.annotate(f"R@10>={tg:g}  alpha={alpha(xs, ys):+.2f}",
                            (xs[-1], ys[-1]), fontsize=7.4, color=c,
                            xytext=(-2, 7), textcoords="offset points",
                            ha="right", va="bottom", fontweight="600")
            ax.set_xscale("log")
            ax.set_yscale("log")
            style(ax, "corpus size N (log)", ylab, title)
            ax.text(.03, .95, "memory profile * alpha rises with the target",
                    transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
            continue
        for p in profs:
            rs = by[p]
            xs, ys = [], []
            for r in rs:
                v = getter(r) if getter else interp_at(r["sweep"], a.target,
                                                       field[kind])
                if v is None:
                    continue
                xs.append(r["n"])
                ys.append(v)
            if not xs:
                continue
            if kind == "depth":
                ax.step(xs, ys, where="post", color=SERIES[p], lw=2.2,
                        ls=DASH[p], marker=MARK[p], ms=5.5, zorder=3)
            else:
                ax.plot(xs, ys, ls=DASH[p], marker=MARK[p], color=SERIES[p],
                        lw=2.2, ms=5.5, zorder=3,
                        mfc="white" if p == "quality" else SERIES[p],
                        mew=1.6)
            al = alpha(xs, ys)
            if al is not None and kind != "depth":
                notes.append(f"{p}: alpha={al:+.2f}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        style(ax, "corpus size N (log)", ylab, title)
        if kind in ("scan", "np", "lat"):
            ax.text(.03, .95, "alpha=0 -> O(1)   alpha=0.5 -> O(sqrtN)   alpha=1 -> O(N)",
                    transform=ax.transAxes, fontsize=7, color=MUTED,
                    va="top")
        if notes:
            ax.text(.03, .04, "   ".join(notes), transform=ax.transAxes,
                    fontsize=8, color=INK2, va="bottom", fontweight="600")
        if kind == "depth":
            ax.set_ylim(-0.4, max(3.4, ax.get_ylim()[1]))

    handles = [Line2D([], [], color=SERIES[p], lw=2.4, ls=DASH[p],
                      marker=MARK[p], ms=5.5,
                      mfc="white" if p == "quality" else SERIES[p], mew=1.6,
                      label=f"{p} profile "
                            f"({by[p][0]['arena_b_per_vec']:.0f} B/vec)")
               for p in profs]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.988, .972),
               frameon=False, ncol=len(profs), fontsize=9.5,
               labelcolor=INK2)

    m = data["meta"]
    fig.text(.045, .955, "AIGVS scalability - HotpotQA", fontsize=17,
             fontweight="700", color=INK)
    fig.text(.045, .915,
             f"{m['n_total']:,} vectors, d={m['d']}, {m['n_queries']:,} "
             f"held-out queries (identical at every scale), exact brute-force "
             f"ground truth * {TARGET(m)} * panels A,B shade light->dark with N",
             fontsize=9, color=INK2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {a.out}")


def TARGET(m):
    return (f"cell size {m['target_cell_size']}, "
            f"{m.get('threads') or 'all'} threads")


if __name__ == "__main__":
    main()
