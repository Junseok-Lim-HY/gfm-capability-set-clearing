#!/usr/bin/env python3
"""Figure 1 for the revised manuscript: four stages, no equations, no sparklines.

    01 physical limit  ->  02 capability set  ->  03 clearing  ->  04 outcomes

Each panel carries one drawing and at most three short labels, so that the
figure remains legible at single-column print size. Writes fig1_overview.pdf
(and .png) next to this file, replacing the pptx export.
"""
from __future__ import annotations

import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

HERE = pathlib.Path(__file__).resolve().parent
DEEP = "#12605F"
INK = "#1B2429"
GREY = "#6B747C"
BOX = "#E4572E"
BAND = "#F4A259"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8})

TEXT_IN = 7.0
fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.30))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

W, H, TOP = 22.0, 24.0, 2.0
LEFTS = [1.5, 26.5, 51.5, 76.5]
TITLES = ("Physical current limit", "Capability set", "Multi-period clearing",
          "Procurement outcomes")
SUB = ("continuous and short-term rating", "reactive current, voltage, band",
       "linear programme, prices as duals", "cost, price, deliverability")

for k, (left, title, sub) in enumerate(zip(LEFTS, TITLES, SUB), 1):
    ax.add_patch(FancyBboxPatch((left, TOP), W, H, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor="white", edgecolor=DEEP, linewidth=1.1, zorder=2))
    ax.add_patch(Circle((left + 2.6, TOP + H - 2.6), 1.6, facecolor=DEEP, edgecolor="white",
                        linewidth=1.5, zorder=4))
    ax.text(left + 2.6, TOP + H - 2.6, str(k), ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold", zorder=5)
    ax.text(left + 5.2, TOP + H - 2.6, title, ha="left", va="center", fontsize=8.6,
            color=DEEP, fontweight="bold", zorder=3)
    ax.text(left + W / 2, TOP + 1.6, sub, ha="center", va="center", fontsize=6.8, color=GREY, zorder=3)
for left in LEFTS[:-1]:
    ax.add_patch(FancyArrowPatch((left + W + 0.4, TOP + H / 2), (left + W + 2.6, TOP + H / 2),
                                 arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=INK, zorder=4))


def panel(left):
    a = fig.add_axes([(left + 1.2) / 100, (TOP + 3.6) / 30, (W - 2.4) / 100, (H - 8.2) / 30])
    a.set_xlim(0, 100); a.set_ylim(0, 60); a.axis("off"); a.set_facecolor("none")
    return a


# 1 --- physical converter
a = panel(LEFTS[0])
for dx, half in ((0, 11), (4.5, 6)):
    a.plot([8 + dx, 8 + dx], [30 - half, 30 + half], color=INK, lw=1.6)
a.plot([12.5, 26], [30, 30], color=INK, lw=1.1)
a.add_patch(FancyBboxPatch((26, 17), 24, 26, boxstyle="square,pad=0", facecolor="white", edgecolor=INK, lw=1.3, zorder=3))
# converter symbol: diagonal from the lower-left to the upper-right corner,
# dc mark (=) in the upper-left triangle on the store side, ac mark (~) in
# the lower-right triangle on the grid side
a.plot([26, 50], [17, 43], color=INK, lw=1.0, zorder=4)
a.text(33.0, 35.5, "=", fontsize=9, ha="center", va="center", color=INK, zorder=4)
a.text(43.0, 24.5, r"$\sim$", fontsize=9, ha="center", va="center", color=INK, zorder=4)
a.plot([50, 60], [30, 30], color=INK, lw=1.1)
a.add_patch(Circle((63.5, 30), 5.2, facecolor="none", edgecolor=INK, lw=1.2))
a.add_patch(Circle((70.5, 30), 5.2, facecolor="none", edgecolor=INK, lw=1.2))
a.plot([75.7, 86], [30, 30], color=INK, lw=1.1)
a.plot([86, 86], [13, 47], color=DEEP, lw=2.6)
a.text(9, 46, "store", fontsize=6.8, ha="center", color=GREY)
a.text(89, 30, "grid", fontsize=6.8, ha="left", va="center", color=DEEP)
a.annotate("", xy=(84, 53), xytext=(56, 53), arrowprops=dict(arrowstyle="->", color=BOX, lw=1.0))
a.text(70, 55.5, r"$I \leq I^{\mathrm{c}},\ I^{\mathrm{s}}$", fontsize=7.4, ha="center", va="bottom", color=BOX)

# 2 --- capability set
a = panel(LEFTS[1])
a.set_xlim(-5, 100); a.set_ylim(-5, 100)
r_c, r_s, pbar = 62.0, 84.0, 62.0
a.add_patch(Wedge((0, 0), r_s, 0, 90, facecolor=BAND, edgecolor="none", alpha=0.55, zorder=1))
a.add_patch(Wedge((0, 0), r_c, 0, 90, facecolor="#DCEBEA", edgecolor="none", zorder=2))
th = np.linspace(0, np.pi / 2, 200)
a.plot(r_c * np.cos(th), r_c * np.sin(th), color=DEEP, lw=1.3, zorder=3)
a.plot(r_s * np.cos(th), r_s * np.sin(th), color=BAND, lw=1.1, zorder=3)
poly = np.linspace(0, np.pi / 2, 7)
a.plot(r_c * np.cos(poly), r_c * np.sin(poly), "o-", ms=2.6, color=DEEP, lw=0.8, zorder=4)
a.plot([0, pbar, pbar], [pbar, pbar, 0], color=BOX, lw=1.5, zorder=5)
a.plot([0, 100], [0, 0], color=INK, lw=0.8); a.plot([0, 0], [0, 100], color=INK, lw=0.8)
a.text(100, -2, "$P$", fontsize=7.5, ha="right", va="top"); a.text(-2, 98, "$Q$", fontsize=7.5, ha="right", va="top")
a.text(pbar + 2, 4, r"$\bar{P}$", fontsize=7.5, color=BOX, ha="left", va="bottom")
a.text(22, 34, "continuous", fontsize=6.4, color=DEEP, ha="center")
a.text(74, 70, "short-term\nband", fontsize=6.4, color="#B36B1E", ha="center", va="center")
a.text(30, 12, r"radius $\propto V$", fontsize=6.3, color=GREY, ha="center")

# 3 --- clearing
a = panel(LEFTS[2])
nodes = {0: (12, 40), 1: (36, 52), 2: (60, 42), 3: (84, 50), 4: (26, 14), 5: (58, 12), 6: (86, 18)}
edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (3, 6), (1, 4)]
for i, j in edges:
    a.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color=INK, lw=1.0, zorder=2)
conv = {0, 2, 6}
for n, (x, y) in nodes.items():
    a.add_patch(Circle((x, y), 4.2, facecolor=DEEP if n in conv else "white", edgecolor=INK, lw=1.0, zorder=3))
a.text(50, 58, "energy + reserve + inertia, per period", fontsize=6.6, ha="center", va="bottom", color=INK)
a.add_patch(Circle((8, 2), 2.4, facecolor="white", edgecolor=INK, lw=0.8))
a.text(12, 2, "synchronous", fontsize=6.3, va="center", color=INK)
a.add_patch(Circle((58, 2), 2.4, facecolor=DEEP, edgecolor=INK, lw=0.8))
a.text(62, 2, "converter", fontsize=6.3, va="center", color=INK)
a.text(50, 30, r"$\lambda,\ \mu^{\mathrm{r}},\ \mu^{\mathrm{h}}$", fontsize=7.2, ha="center", va="center",
       color=BOX, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

# 4 --- outcomes
a = panel(LEFTS[3])
a.set_xlim(0, 100); a.set_ylim(0, 60)
items = [("cost, price", 46), ("ac validation", 32), ("deliverability", 18)]
for label, y in items:
    a.add_patch(FancyBboxPatch((6, y - 5), 60, 10, boxstyle="round,pad=0.3,rounding_size=1.5",
                               facecolor="#F3F6F6", edgecolor=DEEP, lw=0.9))
    a.text(36, y, label, ha="center", va="center", fontsize=7.0, color=INK)
a.text(82, 46, r"$\delta$, ratio", fontsize=6.6, ha="center", va="center", color=BOX)
a.text(82, 32, "8 / 8", fontsize=6.6, ha="center", va="center", color=DEEP)
a.text(82, 18, "N-1", fontsize=6.6, ha="center", va="center", color=DEEP)
a.annotate("", xy=(6, 46), xytext=(-4, 46), arrowprops=dict(arrowstyle="-", color="none"))

# house style (2026-09-22): every label starts with a capital letter
for _ax in fig.get_axes():
    for _t in _ax.texts:
        _s = _t.get_text()
        if _s and _s[0].isalpha() and _s[0].islower():
            _t.set_text(_s[0].upper() + _s[1:])

fig.savefig(HERE / "fig1_overview.pdf")
fig.savefig(HERE / "fig1_overview.png", dpi=220)
print("saved", HERE / "fig1_overview.pdf")
