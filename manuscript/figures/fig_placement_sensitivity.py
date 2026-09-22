#!/usr/bin/env python3
"""Two figures that replace Tables 9 and 10 of the r2 manuscript.

fig_placement     distribution of delta over the randomised converter
                  placements (placement_spread.csv), reference placement marked
fig_sensitivity   tornado chart of delta for the four sensitivity parameters
                  (sensitivity.csv), values printed at the bar ends
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from make_figures import (BOX, COLUMN_IN, CONFIG, LABEL, RESULTS, TEXT_IN,
                          panel_letters, save, tidy)

KEYS = list(CONFIG)


def figure_placement() -> None:
    frame = pd.read_csv(RESULTS / "placement_spread.csv")
    random = frame[frame.placement == "random"]
    ref = frame[frame.placement == "smallest-first"].set_index(["case", "share"])

    fig = plt.figure(figsize=(COLUMN_IN, COLUMN_IN * 0.84))
    ax = fig.add_axes([0.135, 0.215, 0.845, 0.700])
    rng = np.random.default_rng(7)
    counts = []
    for pos, key in enumerate(KEYS):
        part = random[(random.case == key[0]) & (random.share == key[1])]
        feas = part[part.feasible].delta_pct.to_numpy()
        n_inf = int((~part.feasible).sum())
        colour = CONFIG[key]
        ax.scatter(pos + rng.uniform(-0.17, 0.17, feas.size), feas, s=7,
                   color=colour, alpha=0.30, linewidths=0, zorder=2)
        ax.boxplot([feas], positions=[pos], widths=0.52, whis=(0, 100),
                   showfliers=False, zorder=3,
                   medianprops=dict(color=colour, lw=1.8),
                   boxprops=dict(color=colour, lw=1.1),
                   whiskerprops=dict(color=colour, lw=1.0),
                   capprops=dict(color=colour, lw=1.0))
        ax.plot(pos, float(ref.loc[key, "delta_pct"]), marker="D", ms=6.2,
                color=BOX, markerfacecolor="white", markeredgewidth=1.5,
                zorder=5, ls="none")
        counts.append(f"{feas.size} / {n_inf}")
        med = float(np.median(feas))
        ax.text(pos + 0.30, med, f"{med:.2f}", ha="left", va="center",
                fontsize=8.5, color=colour)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABEL[k].replace(", ", "\n") + "\n" + c
                        for k, c in zip(KEYS, counts)], fontsize=9)
    ax.set_ylabel("$\\delta$ (%)")
    ax.set_ylim(-0.05, 2.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.axhline(0.0, color="#555C63", lw=0.7, ls=(0, (4, 3)))
    ax.legend(handles=[Line2D([], [], marker="D", ms=6.2, color=BOX,
                              markerfacecolor="white", markeredgewidth=1.5,
                              ls="none", label="reference placement")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              borderaxespad=0.1)
    tidy(ax)
    save(fig, "fig_placement")


ROWS = [  # axis, label, (lower value, higher value), value format
    ("rocof", "RoCoF limit (Hz/s)", (0.5, 2.0), "{:g}"),
    ("sync_inertia", "$\\mathcal{H}_g$ (s)", (3.0, 6.0), "{:g}"),
    ("storage_hours", "Storage (h)", (1.0, 8.0), "{:g}"),
    ("offer_ratio", "Offer ratio", (0.25, 4.0), "{:g}"),
]
LOW = "#0B6E70"
HIGH = "#F4A259"


def figure_sensitivity() -> None:
    frame = pd.read_csv(RESULTS / "sensitivity.csv")

    def value(key, axis, val):
        if axis == "reference":
            row = frame[(frame.case == key[0]) & (frame.share == key[1])
                        & (frame.axis == "reference")]
        else:
            row = frame[(frame.case == key[0]) & (frame.share == key[1])
                        & (frame.axis == axis) & np.isclose(frame.value, val)]
        return float(row.gap_pct.iloc[0])

    fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.62))
    axes = []
    lefts = [0.145, 0.640]
    bottoms = [0.650, 0.215]
    width, height = 0.335, 0.270
    ypos = np.arange(len(ROWS))[::-1]
    bar_h = 0.34
    for k, key in enumerate(KEYS):
        ax = fig.add_axes([lefts[k % 2], bottoms[k // 2], width, height])
        axes.append(ax)
        base = value(key, "reference", 0.0)
        xmax = base
        for y, (axis, label, (lo, hi), fmt) in zip(ypos, ROWS):
            for offset, val, colour in ((+bar_h / 2, lo, LOW), (-bar_h / 2, hi, HIGH)):
                d = value(key, axis, val)
                ax.barh(y + offset, d - base, left=base, height=bar_h,
                        color=colour, linewidth=0, zorder=3)
                xmax = max(xmax, d)
                text = f"{abs(d) if abs(d) < 0.005 else d:.2f}"
                if d >= base:
                    ax.text(d + 0.06 * max(base, 1.0), y + offset, text,
                            ha="left", va="center", fontsize=8.5)
                else:
                    ax.text(base + 0.06 * max(base, 1.0), y + offset, text,
                            ha="left", va="center", fontsize=8.5)
        ax.axvline(base, color="#555C63", lw=0.9, ls=(0, (4, 3)), zorder=4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r[1]}\n{r[3].format(r[2][0])} / {r[3].format(r[2][1])}"
                            for r in ROWS], fontsize=8.5)
        span = xmax * 1.30 + 0.4
        ax.set_xlim(-0.06 * span, span)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_title(f"{LABEL[key]}: base $\\delta$ = {base:.2f}%", fontsize=9.5,
                     loc="left", pad=3)
        ax.grid(axis="y", visible=False)
        if k // 2 == 1:
            ax.set_xlabel("$\\delta$ (%)")
        tidy(ax)
    fig.legend(handles=[Patch(color=LOW, label="lower parameter value"),
                        Patch(color=HIGH, label="higher parameter value"),
                        Line2D([], [], color="#555C63", lw=0.9, ls=(0, (4, 3)),
                               label="base case")],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    # Panel letters centred below each panel: under the tick labels in the top
    # row, under the axis label in the bottom row.
    for k, (letter, ax) in enumerate(zip("abcd", axes)):
        box = ax.get_position()
        drop = 0.075 if k // 2 == 0 else 0.125
        fig.text(box.x0 + box.width / 2.0, box.y0 - drop, f"({letter})",
                 ha="center", va="top", fontsize=10)
    save(fig, "fig_sensitivity")


if __name__ == "__main__":
    figure_placement()
    figure_sensitivity()
