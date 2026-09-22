# -*- coding: utf-8 -*-
#!/usr/bin/env python3
r"""The loss term really is in the energy balance.

처음에 쓴 검사가 틀렸다. "상수 L 을 손실로 무는 것" 과 "수요를 L 올리는 것"
이 같을 거라고 놓았는데, demand_mw 는 균형에만 들어가는 것이 아니라 예비력과
관성 요구량에도 들어간다. 수요를 올리면 요구량도 같이 오르므로 두 실험이
애초에 다른 실험이었다. 190 달러 차이는 배관이 아니라 그 차이였다.

균형을 직접 본다. 손실 L 을 물렸으면 발전이 공급된 수요보다 정확히 L 만큼
많아야 한다. 애매할 데가 없다.
"""
import dataclasses
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(r"C:\Users\junse\PycharmProjects\SEGAN_GFM")
sys.path.insert(0, str(ROOT / "model"))

from multiperiod import MultiPeriod, build_case, common_demand_peak
from procurement_lp import Capability

settings = Capability(i_short_term=1.5, priority="reactive", soc_threshold=0.20)
base = build_case("rts24", converter_share=0.2,
                  demand_peak_mw=common_demand_peak("rts24"))
case = dataclasses.replace(
    base, conv_injection_pu=np.full(len(base.demand_mw), 0.8))

plain = MultiPeriod(case, settings, "current")
out0 = plain.solve()
size, T = plain.size, plain.T


def surplus(model, outcome):
    """Generation less served demand, per period. Zero without losses."""
    out = []
    for t in range(T):
        made = (float(outcome["pd"][t].sum()) - float(outcome["pc"][t].sum())
                + float(outcome["pg"][t].sum())
                - float(outcome["curt"][t].sum())
                + float(model.injection[t].sum()))
        out.append(made - float(case.demand_mw[t]))
    return np.asarray(out)


base_surplus = surplus(plain, out0)
print(f"손실 없음 : 발전 - 수요 최대 |{np.abs(base_surplus).max():.3e}| MW")
assert np.abs(base_surplus).max() < 1e-6, "손실 없이도 균형이 안 맞습니다"

zero = [(np.zeros(size), 0.0) for _ in range(T)]
out_zero = MultiPeriod(case, settings, "current", losses=zero).solve()
print(f"영        : 비용 {out0['cost']:.6f} vs {out_zero['cost']:.6f}  "
      f"차 {abs(out0['cost'] - out_zero['cost']):.3e}")
assert abs(out0["cost"] - out_zero["cost"]) < 1e-6, "0 손실이 결과를 바꿨습니다"

L = 20.0
const = [(np.zeros(size), L) for _ in range(T)]
model_c = MultiPeriod(case, settings, "current", losses=const)
out_c = model_c.solve()
got = surplus(model_c, out_c)
print(f"상수 {L} MW : 발전 - 수요 = {got.min():.6f} .. {got.max():.6f} MW  "
      f"(L 이어야 함)")
assert np.abs(got - L).max() < 1e-6, \
    f"균형이 손실을 {got.mean():.3f} 만큼만 반영합니다"

# 결정에 걸린 계수: 동기기 출력 1 MW 마다 2 % 를 손실로 문다.
lin = []
for t in range(T):
    row = np.zeros(size)
    row[plain.index["pg"][t]] = 0.02
    lin.append((row, 0.0))
model_l = MultiPeriod(case, settings, "current", losses=lin)
out_l = model_l.solve()
got_l = surplus(model_l, out_l)
want_l = 0.02 * np.asarray([float(out_l["pg"][t].sum()) for t in range(T)])
print(f"선형      : 비용 {out0['cost']:.2f} -> {out_l['cost']:.2f}  "
      f"({100*(out_l['cost']-out0['cost'])/out0['cost']:+.3f} %),  "
      f"균형 잔차 최대 {np.abs(got_l - want_l).max():.3e} MW")
assert np.abs(got_l - want_l).max() < 1e-6, "결정에 걸린 손실이 균형과 안 맞습니다"
assert out_l["cost"] > out0["cost"], "손실을 물렸는데 비용이 안 올랐습니다"

print("\n배관 통과: 0 은 무해하고, 균형이 손실만큼 더 내고, 값이 매겨집니다")
