from ch5_config import *
from docx_fields import NS, visible
from zipfile import ZipFile
from lxml import etree
import pandas as pd
import numpy as np
import sys
import shutil
import json
sys.stdout.reconfigure(encoding='utf8')
checks = []
def check(name, condition, detail=''):
    checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    if not condition: raise AssertionError((name, detail))

scores = pd.read_csv(D / '环境模块13尺度固定指数_人口移类.csv')
components = pd.read_csv(I / '环境指数分量与方向权重_人口移类.csv')
check('Population excluded from biological urban score',
      not ((components.environment_module == 'urbanization') & components.variable.str.startswith('worldpop_')).any())
check('Population included in all biological waterbody-range activity scores',
      int(((components.environment_module == 'human_activity') & components.variable.str.startswith('worldpop_')).sum()) == 26)
check('All site-range-category scores present', len(scores) == 130 * 13 * 3)
family = pd.read_csv(D / '环境模块_生态维度_13尺度全局置换检验.csv')
assoc = pd.read_csv(D / '全部具体响应_偏Spearman_分层FDR.csv')
carrier = pd.read_csv(D / 'Q4_all_frozen_candidate_multiscale_associations.csv')
check('Primary association count preserved', len(assoc) == 16575)
check('All primary family-range cells present', len(family) == 390)
check('Carrier site/range associations complete', len(carrier) == 12558)
check('Family probability bounds', family.range_maxT_p.between(.002, 1).all())
check('Carrier probability bounds', carrier.family_variables_13range_maxT_p.between(.001, 1).all())
ci = carrier.dropna(subset=['bootstrap_ci95_low', 'bootstrap_ci95_high'])
check('Reported carrier confidence intervals ordered', ci.bootstrap_ci95_low.le(ci.bootstrap_ci95_high).all(), len(ci))
for filename in ['原指数基线复现.json', '主分析复算核对.json', 'Q4_复算核对.json',
                 '水质人口归属与复算核对.json', '双响应系统复算核对.json', '敏感性重算完成.json',
                 '子变量分解及水体差异复核.json']:
    data = json.loads((Q / filename).read_text(encoding='utf8'))
    check(filename, data['passed'])
cv = pd.read_csv(D / '代表响应_重复空间阻塞CV_13尺度.csv')
check('Full spatial cross-validation rerun', len(cv) == 1170)
raw = pd.read_csv(D / '模块内预测变量相关聚类映射.csv')
pop = raw[raw.raw_predictor.str.startswith('worldpop_')]
check('Raw-predictor sensitivity uses exclusive activity classification',
      len(pop) > 0 and pop.environment_module.eq('human_activity').all())
taxon = pd.read_csv(O / '13_生物类群×五家族精细分析_2026-07-24/02_类群家族整体尺度检验/生物类群×五家族×三环境模块_13尺度共同置换检验.csv')
check('Taxon-domain downstream analyses rerun', len(taxon) == 2145)
weights = pd.read_csv(I / '环境指标分量与有效权重.csv')
weight_sums = weights.groupby(['water_body_type', 'range_m', 'environment_module']).nominal_weight_before_composite_rescaling.sum()
check('Hierarchical terrain and equal-input activity weights sum to one', np.allclose(weight_sums, 1))

old = SRC / 'manuscript可视化图件/人口密度移类更新_20261006'
retained_data = D / '未受归类影响的背景数据'
retained_data.mkdir(exist_ok=True)
context = SN / 'Context_unchanged'
context.mkdir(exist_ok=True)
for folder in ['GlobalGDP', 'Supporting', '新增GIS英文图件']:
    source = old / folder
    if source.exists(): shutil.copytree(source, context / folder, dirs_exist_ok=True)
for source in (old / '完整更新统计').glob('*.csv'):
    if source.name.startswith(('全球GDP', '水质_', 'API专家')):
        target = retained_data / source.name
        shutil.copy2(source, target)
        check('Unchanged context data hash ' + source.name, sha(source) == sha(target))
if not (old / '完整更新统计').exists():
    manifest = json.loads((O / '07_旧版备份/删除前备份核验清单.json').read_text(encoding='utf8'))
    for target in sorted(retained_data.glob('*.csv')):
        source_name = str(old / '完整更新统计' / target.name)
        matching = [item for item in manifest['entries'] if item['source'] == source_name]
        check('Unchanged context data hash ' + target.name,
              len(matching) == 1 and sha(target) == matching[0]['sha256'])

baseline = I / '历史基线_仅供差值核验'
baseline.mkdir(exist_ok=True)
baseline_sources = {
    '原城市建设含人口_固定指数.csv': SRC / '09_SourceData/环境模块13尺度固定指数.csv',
    '原城市建设含人口_响应维度检验.csv': SRC / '03_Q2_水生生物整体响应/环境模块_生态维度_13尺度全局置换检验.csv',
    '原城市建设含人口_Q4尺度关联.csv': SRC / 'manuscript可视化图件/Fig6_Q4_specific_biological_carriers_2026-08-08/02_source_data/Q4_all_frozen_candidate_multiscale_associations.csv',
}
for name, source in baseline_sources.items():
    target = baseline / name
    if source.exists():
        shutil.copy2(source, target)
        check('Historical baseline snapshot ' + name, sha(source) == sha(target))
    else:
        # Superseded figures may have been removed after verified archival.
        manifest = json.loads((O / '07_旧版备份/删除前备份核验清单.json').read_text(encoding='utf8'))
        matching = [item for item in manifest['entries'] if item['source'] == str(source)]
        check('Historical baseline snapshot ' + name,
              len(matching) == 1 and sha(target) == matching[0]['sha256'])

for label, filename in [('main', 'Manuscript_20261018.docx'), ('supplement', 'Supplementary_materials_20261018.docx')]:
    with ZipFile(O / filename) as archive:
        tree = etree.fromstring(archive.read('word/document.xml'))
        paragraphs = tree.xpath('//w:body//w:p', namespaces=NS)
        text = '\n'.join(visible(p) for p in paragraphs)
        check(label + ' contains no unresolved field placeholders', '{{F' not in text)
        for forbidden in ['population entered only the urban construction score',
                          'Population was excluded from this biological module',
                          'whereas it was confined to urban construction',
                          'frozen analysis workflow', 'Updated figures used']:
            check(label + ' old/internal wording removed: ' + forbidden, forbidden not in text)
        check(label + ' correct population classification stated', 'direct human activity' in text and 'exclusively' in text)
        if label == 'main':
            for index in [5, 6, 7]:
                image = SN / 'Main' / ('Fig' + str(index) + '.png')
                check('Embedded updated figure hash ' + str(index),
                    archive.read('word/media/image' + str(index) + '.png') == image.read_bytes())
        for value in ['as requested', 'for the user', 'we have updated', 'AUTHOR_INPUT_NEEDED']:
            check(label + ' no internal communication phrase ' + value, value not in text)
for item in json.loads((I / '输入文件_SHA256.json').read_text(encoding='utf8')):
    check('Protected input unchanged ' + Path(item['source']).name, sha(item['source']) == item['sha256'])

for index in [5,6,7]:
    svg = SN / 'Main' / ('Fig' + str(index) + '.svg')
    check('Editable SVG text Fig' + str(index), '<text' in svg.read_text(encoding='utf8'))
write_json(Q / '最终数据与正文核验.json', dict(passed=all(c['passed'] for c in checks), checks=checks,
    analysis_counts=dict(primary=len(assoc), family=len(family), carrier=len(carrier), taxon_family=len(taxon), spatial_cv=len(cv))))
pd.DataFrame(checks).to_csv(Q / '最终数据与正文核验.csv', index=False, encoding='utf-8-sig')
print('All', len(checks), 'data, classification, figure and manuscript checks passed.', flush=True)
