#!/usr/bin/env python3
"""Figures for the manuscript, drawn at the size they are printed.

The canvas width is set to the printed width, so a nominal point in the figure
is a point on the page and in-figure text lands at the size it was asked for.
elsarticle in two-column 5p gives a 252 pt column and a 522 pt text block, with
a 10 pt body; figure text is set at 9 pt so it reads level with the body rather
than shouting over it.

Panel letters go at the bottom centre and carry no description; the description
belongs in the caption. Legends sit outside the axes, above and centred,
wherever a legend inside would sit on the data.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.use("Agg")

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "study" / "results"
sys.path.insert(0, str(HERE.parents[1] / "model"))

import capability as cap  # noqa: E402
from multiperiod import HOURS as HOURS_PER_PERIOD  # noqa: E402
from multiperiod import INITIAL_SOC  # noqa: E402

sys.path.insert(0, str(HERE))

# elsarticle, 5p two-column: 252 pt column, 522 pt text block, 10 pt body
COLUMN_IN = 252.0 / 72.27
TEXT_IN = 522.0 / 72.27

# A restrained modern palette: one warm hue for the model under criticism, one
# cool hue for the one proposed, and a muted set for the four configurations.
BOX = "#E4572E"
CIRCLE = "#0B6E70"
BAND = "#F4A259"
CONFIG = {
    ("ieee39", 0.2): "#2E4057",
    ("ieee39", 0.4): "#5C86A8",
    ("rts24", 0.2): "#8B3A62",
    ("rts24", 0.4): "#D08AA8",
}
LABEL = {
    ("ieee39", 0.2): "IEEE 39, 20%",
    ("ieee39", 0.4): "IEEE 39, 40%",
    ("rts24", 0.2): "RTS-24, 20%",
    ("rts24", 0.4): "RTS-24, 40%",
}
MARKER = {("ieee39", 0.2): "o", ("ieee39", 0.4): "s",
          ("rts24", 0.2): "^", ("rts24", 0.4): "D"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # The body is 10 pt and the canvas is the printed width, so 10 pt here is
    # 10 pt on the page. Strokes and markers are scaled up with the type;
    # larger labels beside hairlines would read as a different figure.
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.6,
    "markers.fillstyle": "full",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.4,
    "ytick.major.size": 3.4,
    "axes.grid": True,
    "grid.color": "#D8DCE0",
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": None,
    "text.usetex": False,
})


def panel_letters(fig, axes, y: float) -> None:
    """Letters at the bottom centre of each panel, clear of the axis label.

    Placed in figure coordinates after the axes are positioned, so the letter
    sits below the label rather than on top of it. The description that would
    otherwise go here belongs in the caption.
    """
    for letter, ax in zip("abcdefgh", axes):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2.0, y, f"({letter})",
                 ha="center", va="bottom", fontsize=10)


def _cap(text: str) -> str:
    """Upper-case the first letter of a label; leave math and symbols alone."""
    if not text:
        return text
    first = text[0]
    if first.isalpha() and first.islower():
        return first.upper() + text[1:]
    return text


def capitalise(fig) -> None:
    """Every axis label, title, legend entry, tick label and in-plot text
    starts with a capital letter (house style requested 2026-09-22)."""
    def fix_legend(legend):
        if legend is None:
            return
        for t in legend.get_texts():
            t.set_text(_cap(t.get_text()))

    for ax in fig.get_axes():
        ax.set_xlabel(_cap(ax.get_xlabel()))
        ax.set_ylabel(_cap(ax.get_ylabel()))
        for loc in ("left", "center", "right"):
            if ax.get_title(loc=loc):
                ax.set_title(_cap(ax.get_title(loc=loc)), loc=loc)
        fix_legend(ax.get_legend())
        for t in ax.texts:
            t.set_text(_cap(t.get_text()))
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            if any(l and l[0].isalpha() and l[0].islower() for l in labels):
                axis.set_ticks(axis.get_ticklocs())
                axis.set_ticklabels([_cap(l) for l in labels])
    for legend in fig.legends:
        fix_legend(legend)
    for t in fig.texts:
        t.set_text(_cap(t.get_text()))


def save(fig, stem: str) -> None:
    capitalise(fig)
    path = HERE / f"{stem}.pdf"
    fig.savefig(path)
    fig.savefig(HERE / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {path.name}")


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------
def figure_capability() -> None:
    """The capability set against the box, and what the box leaves out."""
    fig = plt.figure(figsize=(COLUMN_IN, COLUMN_IN * 1.16))
    ax = fig.add_axes([0.170, 0.128, 0.800, 0.672])

    i_cont, i_short = 1.0, 1.5
    angle = np.linspace(0.0, np.pi / 2.0, 400)
    arc = lambda r: (r * np.cos(angle), r * np.sin(angle))

    # the overload band is the annulus between the two current limits, drawn
    # as one closed region rather than two fills that have to cancel
    inner_x, inner_y = arc(i_cont)
    outer_x, outer_y = arc(i_short)
    ax.fill(np.concatenate([outer_x, inner_x[::-1]]),
            np.concatenate([outer_y, inner_y[::-1]]),
            color=BAND, alpha=0.22, linewidth=0, zorder=1)

    for voltage, tone, style, width in ((1.05, "#7768AE", (0, (5, 2)), 1.0),
                                        (0.95, "#7768AE", (0, (1, 2)), 1.0)):
        ax.plot(*arc(voltage * i_cont), color=tone, ls=style, lw=width,
                zorder=3)

    ax.plot(inner_x, inner_y, "-", color=CIRCLE, lw=1.6, zorder=4)
    ax.plot(outer_x, outer_y, "-", color=CIRCLE, lw=1.1, alpha=0.6, zorder=4)
    ax.plot([0.0, 1.0, 1.0], [1.0, 1.0, 0.0], "-", color=BOX, lw=1.7, zorder=5)

    # the box corner sits outside the continuous circle: that is the optimism
    ax.annotate("optimistic in $Q$", xy=(1.0, 1.0), xytext=(1.12, 1.30),
                color=BOX, fontsize=9.5, ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=BOX, lw=0.9))
    ax.annotate("", xy=(1.48, 0.12), xytext=(1.02, 0.12),
                arrowprops=dict(arrowstyle="->", color=CIRCLE, lw=0.9),
                zorder=6)
    ax.text(1.25, 0.20, "conservative\nin $P$", color=CIRCLE, fontsize=9.5,
            ha="center", va="bottom", zorder=6)
    level = 0.34
    ax.annotate("", xy=(np.sqrt(1.05 ** 2 - level ** 2), level),
                xytext=(np.sqrt(0.95 ** 2 - level ** 2), level),
                arrowprops=dict(arrowstyle="<->", color="#7768AE", lw=1.3),
                zorder=6)
    # In the empty interior of the disc, level with the arrow it belongs to
    # and to its left. At (0.72, 0.46) the label sat on the two arcs it names
    # and on the capability boundary, and two lines of type through three
    # curves is unreadable however the colours are chosen. It cannot go much
    # further right: the 0.95 arc curves in as it rises, and the label has to
    # clear it.
    ax.text(0.56, 0.27, "voltage shifts\nthe boundary", color="#7768AE",
            fontsize=9.5, ha="center", va="bottom", zorder=6)

    ax.set_xlim(0.0, 1.62)
    ax.set_ylim(0.0, 1.62)
    ax.set_aspect("equal")
    # The two circles bound different active coordinates -- the continuous one
    # bounds P+R, the short-term one P+R+H -- so naming either on the axis
    # mislabels the other. A generic coordinate is the honest label.
    ax.set_xlabel("Active coordinate $x$ (pu of rating)")
    ax.set_ylabel("Reactive power $Q$ (pu of rating)")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    tidy(ax)

    handles = [
        Line2D([], [], color=BOX, lw=1.7, label="bound, $P+R+H\\leq\\bar{P}$"),
        Line2D([], [], color=CIRCLE, lw=1.6, label="$VI^{\\mathrm{c}}$"),
        Line2D([], [], color=CIRCLE, lw=1.1, alpha=0.6,
               label="$VI^{\\mathrm{s}}$"),
        Line2D([], [], color="#7768AE", lw=1.0, ls=(0, (5, 2)),
               label="$V=1.05$"),
        Line2D([], [], color="#7768AE", lw=1.0, ls=(0, (1, 2)),
               label="$V=0.95$"),
        Patch(facecolor=BAND, alpha=0.22, label="overload band"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.002), handlelength=1.8,
               columnspacing=1.2, handletextpad=0.5)
    save(fig, "fig2_capability")


# --------------------------------------------------------------------------
def figure_sign_flip() -> None:
    """The three omissions, isolated, with their three signatures."""
    frame = pd.read_csv(RESULTS / "sign_flip.csv").dropna(subset=["error_pct"])
    fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.40))
    axes = [fig.add_axes([left, 0.300, 0.253, 0.520])
            for left in (0.075, 0.404, 0.733)]

    panels = (
        ("overload", "Short-term rating $I^{\\mathrm{s}}$ (pu)",
         "Cost difference $\\delta$ (%)"),
        ("voltage", "Terminal voltage $V$ (pu)", None),
        ("reactive", "Reactive duty (pu of zonal load $Q$)", None),
    )
    for ax, (omission, xlabel, ylabel) in zip(axes, panels):
        view = frame[frame.omission == omission]
        for key, colour in CONFIG.items():
            part = view[(view.case == key[0]) & (view.share == key[1])]
            ax.plot(part.value, part.error_pct, marker=MARKER[key],
                    color=colour, label=LABEL[key], markerfacecolor="white",
                    markeredgewidth=1.0)
        ax.axhline(0.0, color="#555C63", lw=0.7, ls=(0, (4, 3)))
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        tidy(ax)

    axes[0].set_xticks([1.0, 1.1, 1.2, 1.35, 1.5])
    axes[1].set_xticks([0.95, 0.98, 1.00, 1.02, 1.05])
    axes[1].set_xticklabels(["0.95", "0.98", "1.00", "1.02", "1.05"])
    axes[2].set_xticks([0.0, 0.35, 0.80, 1.20])
    axes[1].axvline(1.0, color="#555C63", lw=0.7, ls=(0, (1, 2)))
    # Each panel gets the range its own data needs. A common wide range was
    # left over from an earlier sweep and pressed all three signatures onto
    # the zero line; panel (c) was worse than flat, since two curves ran off
    # the bottom. The limits are read from the frame so a re-run cannot
    # silently clip a curve again.
    for ax, (omission, _, _) in zip(axes, panels):
        column = frame[frame.omission == omission].error_pct
        low, high = float(column.min()), float(column.max())
        pad = max(0.08 * (high - low), 0.05)
        ax.set_ylim(min(low - pad, -pad * 0.6), max(high + pad, pad * 0.6))
    # With the panels rescaled the curves now reach the corners the tags used
    # to sit in, so each tag moves to the quadrant its own panel leaves empty:
    # low and right in (a), low and left in (c).
    for ax, tag, tone, x, y, align in (
            (axes[0], "always conservative", CIRCLE, 0.96, 0.06, "right"),
            (axes[2], "always optimistic", BOX, 0.04, 0.06, "left")):
        ax.text(x, y, tag, transform=ax.transAxes, ha=align, va="bottom",
                fontsize=9.5, color=tone)
    axes[1].text(0.5, 0.94, "sign flips at 1.0 pu", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9.5, color="#2E4057")

    handles = [Line2D([], [], color=CONFIG[k], marker=MARKER[k],
                      markerfacecolor="white", markeredgewidth=1.0,
                      label=LABEL[k]) for k in CONFIG]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.8,
               columnspacing=1.6, handletextpad=0.5)
    panel_letters(fig, axes, y=0.095)
    save(fig, "fig4_sign_flip")


# --------------------------------------------------------------------------
def figure_price_and_curtailment() -> None:
    """What the shape costs: the inertia price, and the energy spilled."""
    frame = pd.read_csv(RESULTS / "case_study.csv")
    view = frame[(frame.axis == "dispatch") & frame.feasible]

    fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.43))
    left = fig.add_axes([0.098, 0.300, 0.355, 0.520])
    right = fig.add_axes([0.610, 0.300, 0.360, 0.520])

    for key, colour in CONFIG.items():
        part = view[(view.case == key[0]) & (view.share == key[1])]
        ratio = (part[part.form == "box"].set_index("dispatch").price_inertia
                 / part[part.form == "current"].set_index("dispatch")
                 .price_inertia)
        left.plot(ratio.index, ratio.values, marker=MARKER[key], color=colour,
                  label=LABEL[key], markerfacecolor="white",
                  markeredgewidth=1.0)
    left.axhline(1.0, color="#555C63", lw=0.7, ls=(0, (4, 3)))
    # e_i in the formulation: the exogenous injection the converter
    # arrives with, before the store acts on it and before curtailment.
    # It is not the cleared active power, and calling it "dispatch"
    # invited that reading.
    left.set_xlabel("Available injection $e$ before storage "
                    "and curtailment (pu)")
    left.set_ylabel("Inertia price, box $\\div$ set")
    left.set_yscale("log")
    left.set_yticks([1, 2, 5, 10])
    left.set_yticklabels(["1", "2", "5", "10"])
    left.set_xticks([0.4, 0.6, 0.8, 1.0])
    tidy(left)

    spill = view[view.dispatch == 1.0]
    keys = list(CONFIG)
    positions = np.arange(len(keys))
    box_mwh = [float(spill[(spill.case == k[0]) & (spill.share == k[1])
                           & (spill.form == "box")].curtailed_mwh.iloc[0])
               for k in keys]
    circle_mwh = [float(spill[(spill.case == k[0]) & (spill.share == k[1])
                              & (spill.form == "current")].curtailed_mwh.iloc[0])
                  for k in keys]
    right.bar(positions - 0.19, box_mwh, width=0.36, color=BOX, alpha=0.85,
              edgecolor="white", linewidth=0.5)
    right.bar(positions + 0.19, circle_mwh, width=0.36, color=CIRCLE,
              alpha=0.85, edgecolor="white", linewidth=0.5)
    for x, top, bottom in zip(positions, box_mwh, circle_mwh):
        if top - bottom > 1.0:
            right.annotate(f"$\\Delta$ {top - bottom:,.0f}",
                           xy=(x, max(top, bottom)), xytext=(0, 4),
                           textcoords="offset points", ha="center",
                           fontsize=9.5, color=BOX)
    right.set_xticks(positions)
    right.set_xticklabels([LABEL[k].replace(", ", "\n") for k in keys],
                          fontsize=9.5)
    right.set_ylabel("Energy curtailed over the day (MWh)")
    right.set_ylim(0, max(box_mwh) * 1.22)
    tidy(right)

    handles = [Line2D([], [], color=CONFIG[k], marker=MARKER[k],
                      markerfacecolor="white", markeredgewidth=1.0,
                      label=LABEL[k]) for k in CONFIG]
    handles += [Patch(facecolor=BOX, alpha=0.85, label="bound"),
                Patch(facecolor=CIRCLE, alpha=0.85, label="capability set")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.7,
               columnspacing=1.3, handletextpad=0.5)
    panel_letters(fig, [left, right], y=0.105)
    save(fig, "fig5_price_curtailment")


# --------------------------------------------------------------------------
def figure_network_and_voltage() -> None:
    """The two questions about the network, answered."""
    frame = pd.read_csv(RESULTS / "case_study.csv")
    network = frame[(frame.axis == "network") & frame.feasible]
    history = pd.read_csv(RESULTS / "voltage_history.csv")

    fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.43))
    left = fig.add_axes([0.098, 0.300, 0.355, 0.470])
    right = fig.add_axes([0.615, 0.300, 0.355, 0.470])

    # Two increments, not one total. Adding the branch constraints and the
    # zonal reactive requirement together leaves their contributions
    # unattributable, so each rung is drawn separately and stacked.
    keys = list(CONFIG)
    positions = np.arange(len(keys))
    rungs = (("branches", "single bus, global Q", "dc branches, global Q",
              0.55), ("zonal $Q$", "dc branches, global Q",
                      "dc branches, zonal Q", 0.28))
    for offset, form, colour in ((-0.19, "box", BOX),
                                 (0.19, "current", CIRCLE)):
        bottoms = np.zeros(len(keys))
        for label, base_rung, next_rung, alpha in rungs:
            step = []
            for key in keys:
                part = network[(network.case == key[0])
                               & (network.share == key[1])
                               & (network.form == form)]
                a = float(part[part.rung == base_rung].cost.iloc[0])
                b = float(part[part.rung == next_rung].cost.iloc[0])
                step.append(100.0 * (b - a) / a)
            left.bar(positions + offset, step, bottom=bottoms, width=0.36,
                     color=colour, alpha=alpha, edgecolor="white",
                     linewidth=0.5)
            bottoms = bottoms + np.array(step)
        # One label per bar, on the total, and on every bar. Suppressing the
        # small ones left two of the four configurations with nothing drawn
        # and nothing written, which reads as missing data. Rounding them all
        # to two decimals was no better: it printed 0.00 against a rung that
        # is not zero and against one that is, and the reader cannot tell
        # which is which. Each label carries the digits its own value needs.
        for x, value in zip(positions + offset, bottoms):
            size = 9.0
            if abs(value) >= 0.05:
                text = f"{value:.2f}"
            elif abs(value) > 5e-9:
                text, size = f"{value:.4f}", 7.6
            else:
                text = "0"
            # The two labels of a pair are the same width as the bars they sit
            # on, so stacking both above the axis crowds them. Where the bars
            # are too short to see, the pair is split across the zero line
            # instead: the bound below it, the capability set above.
            #
            # A bar tall enough to carry its own label keeps it, on top, even
            # for the bound. Sending +0.99 below the axis put the number a
            # bar's height away from the bar it describes, and reading it as
            # negative costs more than the tidiness of the rule is worth.
            below = form == "box" and value < 0.30
            anchor = min(value, 0.0) if below else max(value, 0.0)
            left.annotate(text, xy=(x, anchor), xytext=(0, -12 if below else 3),
                          textcoords="offset points", ha="center",
                          va="top" if below else "bottom",
                          fontsize=size, color=colour)
    left.axhline(0.0, color="#555C63", lw=0.7)
    left.set_xticks(positions)
    left.set_xticklabels([LABEL[k].replace(", ", "\n") for k in keys],
                         fontsize=9.5)
    left.set_ylabel("Cost added (%)")
    # 4.2 was set when the tallest rung was expected to be larger. It is 1.18,
    # so three quarters of the panel was empty and the -0.09 rung was a hair
    # above the axis.
    left.set_ylim(-0.62, 1.55)
    tidy(left)

    # Three of the four runs settle at zero to five decimals and lie on top of
    # one another, so a marker on every round drew a solid bar of overlapping
    # symbols and the series could not be told apart. The markers are thinned
    # and staggered by series, and each run is named at the point where it
    # stops, which is the only place the four are guaranteed to be apart.
    for index, (key, colour) in enumerate(CONFIG.items()):
        part = history[(history.case == key[0]) & (history.share == key[1])
                       & (history.form == "current")]
        settled = bool(part.converged.iloc[-1])
        right.plot(part["round"], part.cost_rel, marker=MARKER[key],
                   color=colour, label=LABEL[key], markerfacecolor="white",
                   markeredgewidth=1.0, ls="-" if settled else (0, (2, 2)),
                   markevery=(index, 4), markersize=5.0)
        last_round = int(part["round"].iloc[-1])
        right.annotate(f"{LABEL[key]}  ({last_round})",
                       xy=(last_round, part.cost_rel.iloc[-1]),
                       xytext=(5, 3 if index % 2 == 0 else -11),
                       textcoords="offset points", ha="left", fontsize=8.5,
                       color=colour)
        # A run that stopped at the round cap is marked on the figure, not
        # only in the text: a flat cost line is exactly what a non-converged
        # run looks like, and the two must not be confusable.
        if not settled:
            right.annotate("not converged",
                           xy=(part["round"].iloc[-1], part.cost_rel.iloc[-1]),
                           xytext=(-4, 6), textcoords="offset points",
                           ha="right", fontsize=9.0, color=colour)
    right.axhline(0.0, color="#555C63", lw=0.7, ls=(0, (4, 3)))
    right.set_xlabel("Voltage iteration round")
    right.set_ylabel("Change in cleared cost (%)")
    # Room on the right for the end-of-run labels.
    right.set_xlim(-1.5, float(history["round"].max()) + 21.0)
    tidy(right)

    handles = [Patch(facecolor=BOX, alpha=0.85, label="bound"),
               Patch(facecolor=CIRCLE, alpha=0.85, label="capability set"),
               Patch(facecolor="#7A7F85", alpha=0.55, label="branches"),
               Patch(facecolor="#7A7F85", alpha=0.28, label="zonal $Q$")]
    handles += [Line2D([], [], color=CONFIG[k], marker=MARKER[k],
                       markerfacecolor="white", markeredgewidth=1.0,
                       label=LABEL[k]) for k in CONFIG]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.7,
               columnspacing=1.3, handletextpad=0.5)
    panel_letters(fig, [left, right], y=0.105)
    save(fig, "fig7_network_voltage")





# --------------------------------------------------------------------------
def figure_overview() -> None:
    """Figure 1 is no longer drawn here.

    It is now authored in PowerPoint, in figures/논문그림.pptx, and exported
    from slide 1. Running this would silently replace that export with the
    older matplotlib drawing, so it refuses instead. overview.py is kept
    because it still documents what the panels were built from.
    """
    raise SystemExit(
        "Figure 1 comes from figures/논문그림.pptx, not from overview.py. "
        "Export slide 1 to fig1_overview.pdf instead of running this.")
    import overview

    overview.draw(RESULTS, TEXT_IN, save)


# --------------------------------------------------------------------------
def figure_sign_map() -> None:
    """The three regions of the Proposition over voltage and rating.

    The crossing exists only where the reachable radius exceeds the
    active-power bound. Where it does not there is nothing to contour, and
    shading that region as though the crossing were zero is what the earlier
    version of this figure did; it read as agreement where the bound is in
    fact overstating.
    """
    fig = plt.figure(figsize=(COLUMN_IN, COLUMN_IN * 1.02))
    ax = fig.add_axes([0.180, 0.145, 0.600, 0.760])

    voltage = np.linspace(0.90, 1.10, 500)
    i_short = np.linspace(1.0, 1.8, 500)
    V, I = np.meshgrid(voltage, i_short)
    reach = V * I                       # L / S at full band availability
    crossing = np.where(reach > 1.0,
                        np.sqrt(np.clip(reach ** 2 - 1.0, 0.0, None)), np.nan)

    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    shading = ax.contourf(V, I, crossing, levels=levels, cmap="BuPu",
                          alpha=0.9, extend="max")
    lines = ax.contour(V, I, crossing, levels=levels[1:], colors="white",
                       linewidths=0.5)
    ax.clabel(lines, fmt="%.1f", fontsize=8.0, inline=True)

    # region (iii) has no crossing to draw, so it is hatched rather than
    # shaded at zero, which would read as agreement
    ax.contourf(V, I, np.where(reach <= 1.0, 1.0, np.nan), levels=[0.5, 1.5],
                colors="none", hatches=["///"], zorder=2)
    ax.contour(V, I, reach, levels=[1.0], colors=[BOX], linewidths=1.8,
               zorder=3)

    ax.plot([1.0], [1.0], marker="*", markersize=9, color=BOX,
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.annotate("$L=\\bar{P}$: the only\npoint of agreement", xy=(1.0, 1.0),
                xytext=(1.015, 1.20), fontsize=8.5, color=BOX, zorder=5,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=BOX, lw=0.8))
    ax.text(0.905, 1.40, "region (iii): $L<\\bar{P}$\nno crossing, and the\n"
            "bound overstates at\nevery $\\tilde{Q}$", fontsize=8.5, color=BOX,
            ha="left", va="top", zorder=5, linespacing=1.4)

    position = ax.get_position()
    cax = fig.add_axes([position.x1 + 0.030, position.y0, 0.042,
                        position.height])
    bar = fig.colorbar(shading, cax=cax)
    bar.set_label("$\\tilde{Q}^{\\mathrm{crit}}$ (pu of rating)", fontsize=10)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_linewidth(0.5)

    ax.set_xlabel("Terminal voltage $V$ (pu)")
    ax.set_ylabel("Short-term rating $I^{\\mathrm{s}}$ (pu)")
    ax.set_xticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax.grid(False)
    save(fig, "fig3_sign_map")


def figure_day() -> None:
    """Where in the day the two formulations part company."""
    frame = pd.read_csv(RESULTS / "day_schedule.csv")
    fig = plt.figure(figsize=(TEXT_IN, TEXT_IN * 0.43))
    left = fig.add_axes([0.098, 0.300, 0.355, 0.520])
    right = fig.add_axes([0.615, 0.300, 0.355, 0.520])

    box = frame[frame.form == "box"].sort_values("period")
    circle = frame[frame.form == "current"].sort_values("period")
    hours = box.hour.to_numpy()

    # A post-step drawn on the interval's left edges stops at the start of the
    # last interval, leaving the final three hours off the plot. Repeating the
    # last value at the closing edge draws the day.
    def whole_day(series):
        values = series.to_numpy()
        return np.append(values, values[-1])

    edges = np.append(hours, hours[-1] + HOURS_PER_PERIOD)

    # The line is one optimum; the band is every optimum. Under the bound the
    # cost-optimal face is wide in reserve, so a line alone would report the
    # tie-break rather than the formulation. Under the capability set the face
    # is a single value and the band collapses onto the line, which is the
    # comparison the panel is making.
    for frm, colour in ((box, BOX), (circle, CIRCLE)):
        left.fill_between(edges, whole_day(frm.reserve_lo_mw),
                          whole_day(frm.reserve_hi_mw), step="post",
                          color=colour, alpha=0.16, linewidth=0)
    left.step(edges, whole_day(box.reserve_mw), where="post", color=BOX,
              lw=1.5)
    left.step(edges, whole_day(circle.reserve_mw), where="post", color=CIRCLE,
              lw=1.5)
    # Demand is drawn for shape, so the factor it was divided by has to be on
    # the plot or the curve is decoration.
    factor = box.demand_mw.max() / circle.reserve_mw.max()
    left.step(edges, whole_day(box.demand_mw / factor), where="post",
              color="#9AA3AB", lw=1.0, ls=(0, (4, 3)))
    left.set_xlabel("Hour of day")
    left.set_ylabel("Converter reserve held (MW)")
    left.set_xticks([0, 6, 12, 18, 24])
    tidy(left)

    # The state of charge is a state, not a rate: the value the clearing
    # returns for period t is where the store ends that interval, so it
    # belongs at the interval's right-hand edge. Plotting it at the left edge
    # hides the opening state entirely and shifts the whole trajectory three
    # hours early.
    #
    # The two trajectories coincide to the width of the line, so the second is
    # drawn dashed over the first. Plotting them as two solid curves would show
    # one curve and leave the reader to guess whether a series was lost.
    edges = np.concatenate([[0.0], hours + HOURS_PER_PERIOD])
    for series, colour, style in ((box.mean_soc, BOX, "-"),
                                  (circle.mean_soc, CIRCLE, (0, (3, 3)))):
        right.plot(edges, np.concatenate([[INITIAL_SOC], series.to_numpy()]),
                   color=colour, lw=1.5, ls=style, marker="o", markersize=2.6,
                   markerfacecolor="white", markeredgewidth=0.9)
    right.axhline(0.20, color="#555C63", lw=0.7, ls=(0, (1, 2)))
    right.axhline(0.05, color="#555C63", lw=0.7, ls=(0, (1, 2)))
    right.text(23.4, 0.215, "full band", fontsize=9.5, color="#555C63",
               ha="right", va="bottom")
    right.text(23.4, 0.065, "energy floor", fontsize=9.5, color="#555C63",
               ha="right", va="bottom")
    right.set_xlabel("Hour of day")
    right.set_ylabel("Fleet mean state of charge")
    right.set_xticks([0, 6, 12, 18, 24])
    right.set_ylim(0.0, 0.85)
    tidy(right)

    # The shaded band had no legend entry, so a reader had to infer what it
    # meant from the caption. It is named here, next to the line it belongs
    # with, because the difference between the two is the point of the panel.
    handles = [
        Line2D([], [], color=BOX, lw=1.5, label="bound"),
        Line2D([], [], color=CIRCLE, lw=1.5, label="capability set"),
        Patch(facecolor="#7A7F85", alpha=0.28,
              label="all equal-cost optima"),
        Line2D([], [], color="#9AA3AB", lw=1.0, ls=(0, (4, 3)),
               label=f"demand / {factor:.1f}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5)
    panel_letters(fig, [left, right], y=0.105)
    save(fig, "fig6_day")


def figure_dynamic() -> None:
    """What a converter does with a schedule the capability set admitted.

    The first two panels are the mechanism and the third is the claim. On the
    left, one operating point through one event on three grids: the response
    builds, then settles, and where it settles depends on how far the terminal
    voltage moved. In the middle, the current that settling is competing for,
    split into the two axes the priority rule allocates between, with the
    instant the limiter binds marked -- without it the settled level in the
    left panel has to be taken on trust. On the right, the whole sweep against
    that movement. The points fall on one curve, which is the point -- the
    shortfall is not a dynamic effect to be simulated case by case, it is the
    constraint being evaluated at a voltage that is not the one that obtains.
    """
    frame = pd.read_csv(RESULTS / "dynamic_check.csv")
    traces = pd.read_csv(RESULTS / "dynamic_traces.csv")

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, TEXT_IN / 2.7))
    left, middle, right = axes

    grids = ["strong", "ordinary", "weak"]
    shade = {"strong": "#2E4057", "ordinary": "#5C86A8", "weak": CIRCLE}

    def trace_of(q_case: str):
        return ((traces.v_pre == 1.00) & (traces.q_case == q_case)
                & (traces.i_short == 1.1) & (traces.priority == "reactive")
                & (traces.reactive_mode == "hold schedule")
                & (traces.voltage == "grid voltage holds"))

    pick = trace_of("low")
    row = frame[(frame.v_pre == 1.00) & (frame.q_case == "low")
                & (frame.i_short == 1.1) & (frame.priority == "reactive")
                & (frame.reactive_mode == "hold schedule")
                & (frame.voltage == "grid voltage holds")]
    scheduled = 0.60 + 0.10 + float(row.h_scheduled.iloc[0])

    left.axhline(scheduled, color=BOX, linestyle="--", linewidth=1.4,
                 label="scheduled", zorder=1)
    for name in grids:
        part = traces[pick & (traces.grid == name)].sort_values("t")
        left.plot(part.t, part.p_out, color=shade[name], label=name, zorder=2)
    left.set_xlim(0.4, 1.48)
    left.set_ylim(0.55, 1.16)
    left.set_xlabel("Time (s)")
    left.set_ylabel("Active power (pu of rating)")
    # Inside the axes, low and right, where the trace has already levelled off
    # and there is nothing to sit on. Three panels leave no room above.
    left.legend(loc="lower right", frameon=False, handlelength=1.6,
                fontsize=8, labelspacing=0.3, borderaxespad=0.4)
    tidy(left)

    # Why it settles where it does. The left panel shows the active power
    # arriving at a level; this shows the current that level is competing for.
    # The priority rule fills the reactive axis first and the active axis gets
    # what the limit leaves, so a reader who cannot see the split has to take
    # the settled value on trust. The axis currents are exact rather than
    # inferred: the model defines p_out = i_d v and q_out = i_q v. The large
    # reactive schedule is the one drawn, because at the small one the two
    # axes do not separate and the panel would show nothing.
    heavy = traces[trace_of("high") & (traces.grid == "weak")].sort_values("t")
    middle.plot(heavy.t, heavy.limit, color="#9AA3AB", linewidth=2.0,
                alpha=0.7, solid_capstyle="butt", label="limit", zorder=1)
    middle.plot(heavy.t, heavy.i_delivered, color=shade["weak"], linewidth=1.5,
                label="$|I|$", zorder=4)
    middle.plot(heavy.t, heavy.p_out / heavy.v_term, color="#2E4057",
                linestyle=(0, (5, 2)), linewidth=1.3, label="active axis",
                zorder=3)
    middle.plot(heavy.t, heavy.q_out / heavy.v_term, color="#C1666B",
                linestyle=(0, (1, 1.6)), linewidth=1.5, label="reactive axis",
                zorder=3)
    held = heavy[heavy.clipped]
    if len(held):
        entry = float(held.t.iloc[0])
        # The marker line stops above the legend, which sits in the lower
        # left corner clear of the reactive-axis trace.
        middle.axvline(entry, ymin=0.40, color="#9AA3AB", linewidth=0.9,
                       linestyle=(0, (2, 2)), zorder=1)
        middle.annotate("limiter binds", xy=(entry, 1.29),
                        xytext=(entry + 0.03, 1.29), fontsize=8,
                        color="#5A6169", va="top")
    middle.set_xlim(0.4, 1.48)
    middle.set_ylim(0.0, 1.35)
    middle.set_xlabel("Time (s)")
    middle.set_ylabel("Current (pu of rating)")
    middle.legend(loc="lower left", frameon=False, handlelength=1.7,
                  fontsize=8, labelspacing=0.3, borderaxespad=0.4)
    tidy(middle)

    floor = 0.0
    marks = {"hold schedule": "o", "voltage droop": "^"}
    for mode, marker in marks.items():
        part = frame[frame.reactive_mode == mode]
        share = 100.0 * part.settled_frac
        # Points below the frame are drawn on it, hollow and at the edge, so
        # that they are visible as excluded rather than quietly dropped. How
        # far below they go is in the text.
        right.scatter(part.v_settled_drop_pct, np.maximum(share, floor + 1.5),
                      s=15, marker=marker, facecolors="none", linewidths=0.9,
                      edgecolors=[shade[g] for g in part.grid], label=mode)
        off = part[share < floor]
        if len(off):
            right.scatter(off.v_settled_drop_pct,
                          np.full(len(off), floor + 1.5), s=34, marker="v",
                          color=BOX, zorder=3)
    right.set_ylim(floor, 112.0)
    right.set_xlabel("Terminal voltage below pre-event (%)")
    right.set_ylabel("Delivered (% of schedule)")
    right.legend(loc="upper right", frameon=False, handlelength=1.2,
                 fontsize=8, labelspacing=0.3, borderaxespad=0.4,
                 scatterpoints=1)
    tidy(right)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    panel_letters(fig, axes, 0.005)
    save(fig, "fig_dynamic")


def main() -> None:
    print("figures:")
    # figure_overview() -- Fig. 1 is exported from the pptx, see above
    figure_capability()
    figure_sign_map()
    figure_sign_flip()
    figure_price_and_curtailment()
    figure_day()
    figure_network_and_voltage()
    figure_dynamic()


if __name__ == "__main__":
    main()
