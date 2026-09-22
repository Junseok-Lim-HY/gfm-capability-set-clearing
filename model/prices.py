#!/usr/bin/env python3
"""One way of turning requirement duals into a reported price.

The inertia requirement is now written once per credible outage, so a period
carries as many rows as there are synchronous machines and all but the binding
one are slack. Averaging over every row therefore divides the price by the
number of machines and reports a number that falls as the fleet grows, which
is an artefact of the row count rather than anything about the system.

What a period costs is the sum of its duals: tightening the requirement by a
megawatt costs the sum of the marginal costs of the rows that bind. Periods
are then averaged, because the day has eight of them and one number is wanted.

Reserve and reactive rows are one per period and one per zone respectively, so
the same rule reduces to the obvious thing for them.
"""
from __future__ import annotations

import collections
import re

import numpy as np

PERIOD = re.compile(r"_t(\d+)")


def period_of(key: str) -> int:
    found = PERIOD.search(key)
    return int(found.group(1)) if found else -1


def price(prices: dict, kind: str, hours: float = 3.0) -> float:
    """Per-period sum of the duals of one requirement, averaged over the day.

    The objective carries the period length, so a dual is a cost per megawatt
    *per period*. Dividing by the period length reports it per megawatt-hour,
    which is the unit the offers are quoted in and the one a reader expects.
    """
    by_period: dict[int, float] = collections.defaultdict(float)
    for key, value in prices.items():
        if key.startswith(kind):
            by_period[period_of(key)] += float(value)
    if not by_period:
        return float("nan")
    return float(np.mean(list(by_period.values()))) / hours
