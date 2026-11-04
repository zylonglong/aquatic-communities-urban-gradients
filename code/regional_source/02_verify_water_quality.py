from ch5_config import *
import sys
import shutil
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf8')
m = load_module('revision_water_core', S / 'ch5_water_core.py')
site = pd.read_csv(WQ / '01_INPUT与过程数据/WQ_Site_Special_13Ranges.csv', low_memory=False)
site.to_pickle(D / '冻结水质Site输入.pkl')
scores, loadings, transforms, modules = m.build_gradients(site)
for (label, r), columns in modules.items():
    if label == '城市化建设':
        assert not any('worldpop' in c for c in columns)
    if label == '人类直接活动':
        assert any('worldpop' in c for c in columns)
oldscore = pd.read_csv(WQ / '02_统计结果/环境模块梯度得分_13尺度.csv')
keys = ['spatial_id', 'water_body_type', 'driver', 'range_m']
check = scores.merge(oldscore, on=keys, suffixes=('_new', '_old'), validate='1:1')
error = float((check.gradient_score_new - check.gradient_score_old).abs().max())
assert len(check) == len(scores) and error < 1e-10
print('Water-quality PCA gradients independently reproduced.', flush=True)

# Keep the original outcome order to reproduce the original random-number stream.
rows = []
for wt, outcome in [('Lake', 'wq_class_score_mean'), ('River', 'wq_class_score_mean'), ('River', 'pollution_index_mean')]:
    raw = site[site.spatial_waterbody_type.eq(wt)][['station_id', 'spatial_id', outcome]].dropna()
    y = raw.groupby('spatial_id', as_index=False).agg(**{outcome: (outcome, 'mean')})
    for driver in ['城市化建设', '人类直接活动', '地形地貌']:
        for r in RANGES:
            g = scores[(scores.water_body_type == wt) & (scores.driver == driver) & (scores.range_m == r)]
            frame = y.merge(g[['spatial_id', 'gradient_score']], on='spatial_id', how='inner')
            rho, p = m.permutation_spearman(frame.gradient_score, frame[outcome])
            low, high = m.bootstrap_spearman(frame.gradient_score, frame[outcome])
            rows.append(dict(water_body_type=wt, response=outcome, driver=driver, range_m=r,
                             spearman_rho=rho, permutation_p=p, ci95_low=low, ci95_high=high))
        print('Verified', wt, outcome, driver, flush=True)
fresh = pd.DataFrame(rows)
fresh['fdr_q'] = fresh.groupby(['water_body_type', 'response', 'driver']).permutation_p.transform(m.bh_q)
old = pd.read_csv(SRC / '02_Q1_水质等级参照/水质等级_三类环境梯度_13尺度响应.csv')
comparison = fresh[fresh.response == 'wq_class_score_mean'].merge(
    old, on=['water_body_type', 'response', 'driver', 'range_m'], suffixes=('_new', '_old'), validate='1:1')
errors = {name: float((comparison[name + '_new'] - comparison[name + '_old']).abs().max())
          for name in ['spearman_rho', 'permutation_p', 'fdr_q', 'ci95_low', 'ci95_high']}
assert len(comparison) == 78 and max(errors.values()) < 1e-9, errors
main = old.copy()
main['environment_module'] = main.driver.map({'地形地貌': 'terrain', '城市化建设': 'urbanization', '人类直接活动': 'human_activity'})
main.to_csv(D / '水质等级_三类环境梯度_13尺度响应.csv', index=False, encoding='utf-8-sig')
comparison.to_csv(D / '水质主结果新旧逐项比较.csv', index=False, encoding='utf-8-sig')
fresh.to_csv(D / '水质117项独立复算.csv', index=False, encoding='utf-8-sig')
dest = SN / 'Q1/02_source_data'
dest.mkdir(parents=True, exist_ok=True)
main.to_csv(dest / 'Figure02_Q1_water_quality_scale.csv', index=False, encoding='utf-8-sig')
retained = []
for source in (WQ / '02_统计结果').glob('*.csv'):
    target = D / 'Q1_水质重算/原始归类正确_核验保留' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    retained.append(dict(source=str(source), sha256=sha(source), unchanged=True))
write_json(Q / '水质人口归属与复算核对.json', dict(passed=True, population_already_in_activity=True,
    gradient_max_error=error, statistic_max_errors=errors, main_rows=78,
    rerun_outcome_range_combinations=len(fresh), permutations=4999, bootstraps=999,
    retained_unchanged_individual_and_sensitivity_outputs=retained))
print('Water-quality coefficient, permutation, FDR and CI invariance verified.', flush=True)
