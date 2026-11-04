# -*- coding: utf-8 -*-
"""Compare water-quality associations before/after nine section corrections."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent
RES = OUT / "02_统计结果"
QA = OUT / "06_QA"
OLD = QA / "9个River断面修缮前结果备查"

keys = ["water_body_type", "response", "driver", "range_m"]
old = pd.read_csv(OLD / "环境模块梯度与水质响应_13尺度.csv")
new = pd.read_csv(RES / "环境模块梯度与水质响应_13尺度.csv")
cols = keys + ["n_spatial_units", "spearman_rho", "ci95_low", "ci95_high", "permutation_p", "fdr_q"]
cmp = old[cols].merge(new[cols], on=keys, suffixes=("_before", "_after"), validate="1:1")
cmp["delta_rho_after_minus_before"] = cmp.spearman_rho_after - cmp.spearman_rho_before
cmp["fdr_significant_before"] = cmp.fdr_q_before < 0.05
cmp["fdr_significant_after"] = cmp.fdr_q_after < 0.05
cmp.to_csv(QA / "9个River断面修缮前后_13尺度响应比较.csv", index=False, encoding="utf-8-sig")

def strongest(d, suffix):
    return (d.assign(abs_rho=d[f"spearman_rho_{suffix}"].abs())
            .sort_values(["water_body_type", "response", "driver", "abs_rho"],
                         ascending=[True, True, True, False])
            .groupby(["water_body_type", "response", "driver"], as_index=False).head(1))

b = strongest(cmp, "before")[[*keys, "spearman_rho_before", "fdr_q_before"]]
a = strongest(cmp, "after")[[*keys, "spearman_rho_after", "fdr_q_after"]]
best = b.merge(a, on=["water_body_type", "response", "driver"], suffixes=("_before", "_after"))
best.to_csv(QA / "9个River断面修缮前后_候选最强尺度比较.csv", index=False, encoding="utf-8-sig")
print(best.to_string(index=False))
