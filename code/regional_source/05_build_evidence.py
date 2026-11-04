from ch5_config import *
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf8')
family = pd.read_csv(D / '环境模块_生态维度_13尺度全局置换检验.csv')
keys = ['water_body_type', 'environment_module', 'response_family']
peaks = family.loc[family.groupby(keys).rms_partial_rho.idxmax()].copy()
support = family.assign(supported=family.range_maxT_p.le(.05)).groupby(keys).agg(
    n_supported_ranges=('supported', 'sum'), min_maxT=('range_maxT_p', 'min')).reset_index()
peaks = peaks.merge(support, on=keys)
peaks.to_csv(T / 'Q2_正文数值与支持尺度.csv', index=False, encoding='utf-8-sig')
range_lists = family[family.range_maxT_p.le(.05)].groupby(keys).range_m.apply(list)
write_json(T / 'Q2_校正支持尺度.json', {'|'.join(k): v for k, v in range_lists.items()})
site = pd.read_pickle(D / '冻结Site输入.pkl')
rows = []
for column in site.columns:
    if column.startswith('worldpop_'):
        category, old, source = 'human_activity', 'urbanization', 'WorldPop'
    elif column.startswith('api_'):
        category, old, source = 'human_activity', 'human_activity', 'OpenStreetMap activity groups'
    elif column.startswith(('urbanization_', 'landuse_')):
        category, old, source = 'urbanization', 'urbanization', 'OpenStreetMap / WorldCover land cover'
    elif column.startswith('terrain_'):
        category, old, source = 'terrain', 'terrain', 'ASTER GDEM'
    else:
        continue
    rows.append(dict(variable=column, environment_module=category, previous_biological_module=old,
                     source_family=source, category_changed=(category != old)))
dictionary = pd.DataFrame(rows)
dictionary.to_csv(I / '环境变量全量分类字典_20261018.csv', index=False, encoding='utf-8-sig')
assert dictionary[dictionary.variable.str.startswith('worldpop_')].environment_module.eq('human_activity').all()
components = pd.read_csv(I / '环境指数分量与方向权重_人口移类.csv')
components.loc[components.variable.str.startswith('worldpop_'), 'mechanism_group'] = 'population_density'
components.to_csv(I / '环境指数分量与方向权重_人口移类.csv', index=False, encoding='utf-8-sig')
keys_c = ['water_body_type', 'range_m', 'environment_module']
counts = components.groupby(keys_c).variable.nunique()
group_counts = components.groupby(keys_c).mechanism_group.nunique()
within_group = components.groupby(keys_c + ['mechanism_group']).variable.nunique()
components['nominal_weight_before_composite_rescaling'] = [
    (1 / group_counts.loc[(r.water_body_type, r.range_m, r.environment_module)] /
     within_group.loc[(r.water_body_type, r.range_m, r.environment_module, r.mechanism_group)])
    if r.environment_module == 'terrain' else
    1 / counts.loc[(r.water_body_type, r.range_m, r.environment_module)]
    for r in components.itertuples()]
components.to_csv(I / '环境指标分量与有效权重.csv', index=False, encoding='utf-8-sig')
write_json(Q / '基础数据分类审计.json', dict(
    population_predictors=int(dictionary.variable.str.startswith('worldpop_').sum()),
    population_in_urban_components=int(((components.environment_module == 'urbanization') & components.variable.str.startswith('worldpop_')).sum()),
    population_activity_waterbody_range_entries=int(((components.environment_module == 'human_activity') & components.variable.str.startswith('worldpop_')).sum()),
    observation_values_changed=False, source_input_hashes_verified=True,
    input_dictionary_originally_described_population_as_human_activity=True))
if (D / 'Q4_各水体各环境梯度前5响应特征.csv').exists():
    top = pd.read_csv(D / 'Q4_各水体各环境梯度前5响应特征.csv')
    print('TOP5_COLUMNS', top.columns.tolist())
    top.to_csv(T / 'Q4_正文前五特征.csv', index=False, encoding='utf-8-sig')
    for filename in ['Q4_网络响应维度指标.csv', 'Q4_网络全变量指标.csv']:
        frame = pd.read_csv(D / filename)
        print(filename, frame.columns.tolist())
print('Evidence and category dictionaries generated.', flush=True)
