#!/usr/bin/env python3
"""Network data for the clearing: shift factors, branch limits, zones.

The clearing in `multiperiod.py` began as a single-bus problem with converter
terminal voltages borrowed from a power flow. That leaves two things a referee
will ask about. Branch limits do not appear at all, so the clearing can
schedule a dispatch the network cannot carry; and the reactive requirement is
imposed on the system as a whole, when reactive power does not travel far
enough for that to mean anything. This module supplies what is needed to close
both: a dc shift-factor matrix with branch limits, and a zone partition for the
reactive requirement.

Nothing here decides anything. It reads a pandapower case and returns arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp
from pandapower.pypower.makePTDF import makePTDF

# pypower branch column for the long-term rating, in MVA
RATE_A = 5


@dataclass(frozen=True)
class Network:
    """Everything the clearing needs to know about the wires."""

    ptdf: np.ndarray          # (n_branch, n_bus), flow per unit of injection
    limit_mw: np.ndarray      # (n_branch,), long-term branch rating
    branch_name: list         # (n_branch,), for reporting which one binds
    bus_of: dict              # pandapower bus index -> column of ptdf
    zone_of_bus: np.ndarray   # (n_bus,), zone label per column
    zones: np.ndarray         # sorted unique zone labels


def extract(net) -> Network:
    """Read shift factors, ratings and zones out of a solved pandapower case.

    The dc power flow is run only to populate the internal ppc; its solution is
    discarded. Shift factors are taken about pandapower's own slack, which is
    the reference the case was built with.
    """
    pp.rundcpp(net)
    ppc, lookup = net._ppc, net._pd2ppc_lookups

    ptdf = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    limit = np.asarray(ppc["branch"][:, RATE_A], dtype=float).real

    # A zero rating in the ppc means "not given", not "no capacity". Rather
    # than invent a number, those branches are dropped from the constraint set
    # and reported, so the omission is visible instead of silent.
    given = limit > 0.0
    if not given.all():
        dropped = int((~given).sum())
        print(f"  note: {dropped} of {len(limit)} branches carry no rating "
              f"and are left unconstrained")

    names = []
    for kind, (start, stop) in lookup["branch"].items():
        table = getattr(net, kind)
        for offset, index in enumerate(table.index):
            if start + offset < stop:
                names.append(f"{kind}{index}")
    while len(names) < len(limit):
        names.append(f"branch{len(names)}")

    bus_of = {int(b): int(lookup["bus"][b]) for b in net.bus.index}
    n_bus = ppc["bus"].shape[0]

    zone_of_bus = np.zeros(n_bus, dtype=float)
    for b in net.bus.index:
        zone_of_bus[bus_of[b]] = float(net.bus.at[b, "zone"])

    return Network(
        ptdf=ptdf[given][:, :n_bus],
        limit_mw=limit[given],
        branch_name=[n for n, keep in zip(names, given) if keep],
        bus_of=bus_of,
        zone_of_bus=zone_of_bus,
        zones=np.unique(zone_of_bus),
    )


def nodal_load(net, scale: float) -> np.ndarray:
    """Active and reactive load per ppc bus column at a given load scale."""
    lookup = net._pd2ppc_lookups["bus"]
    n_bus = net._ppc["bus"].shape[0]
    p = np.zeros(n_bus)
    q = np.zeros(n_bus)
    for index, row in net.load.iterrows():
        column = int(lookup[int(row.bus)])
        p[column] += float(row.p_mw) * scale
        q[column] += float(row.q_mvar) * scale
    return p, q
