from ch5_config import *
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import numpy as np
import shutil
import sys
sys.stdout.reconfigure(encoding='utf8')

# Make baseline comparisons independent of the figure folders being retired.
script = S / 'ch5_primary_analysis.py'
text = script.read_text(encoding='utf8').replace(
    "SRC/'09_SourceData/环境模块13尺度固定指数.csv'",
    "I/'历史基线_仅供差值核验/原城市建设含人口_固定指数.csv'").replace(
    "SRC/'03_Q2_水生生物整体响应/环境模块_生态维度_13尺度全局置换检验.csv'",
    "I/'历史基线_仅供差值核验/原城市建设含人口_响应维度检验.csv'")
script.write_text(text, encoding='utf8')
script = S / 'ch5_carrier_analysis.py'
text = script.read_text(encoding='utf8').replace(
    "SRC/'manuscript可视化图件/Fig6_Q4_specific_biological_carriers_2026-08-08/02_source_data/Q4_all_frozen_candidate_multiscale_associations.csv'",
    "I/'历史基线_仅供差值核验/原城市建设含人口_Q4尺度关联.csv'")
script.write_text(text, encoding='utf8')
frozen = S / '冻结绘图依赖'
frozen.mkdir(exist_ok=True)
for source in (S / '绘图适配').glob('*.py'):
    if source.name.startswith(('q4_', 'fig5b_')):
        shutil.copy2(source, frozen / source.name)
script = S / 'ch5_network_data.py'
text = script.read_text(encoding='utf8').replace("ROOT/'src/modeling'/name", "S/'冻结绘图依赖'/name")
script.write_text(text, encoding='utf8')

old = pd.read_csv(I / '历史基线_仅供差值核验/原城市建设含人口_响应维度检验.csv')
new = pd.read_csv(D / '环境模块_生态维度_13尺度全局置换检验.csv')
keys = ['water_body_type', 'environment_module', 'response_family', 'range_m']
comparison = new.merge(old, on=keys, suffixes=('_new', '_old'), validate='1:1')
comparison.to_csv(T / '人口移类前后_响应维度全量比较.csv', index=False, encoding='utf-8-sig')
support = comparison.groupby(['water_body_type','environment_module']).apply(
    lambda g: pd.Series(dict(old_supported=int(g.range_maxT_p_old.le(.05).sum()),
                           new_supported=int(g.range_maxT_p_new.le(.05).sum()),
                           max_rms_change=float((g.rms_partial_rho_new-g.rms_partial_rho_old).abs().max()))),
    include_groups=False).reset_index()
support.to_csv(T / '人口移类前后_支持单元汇总.csv', index=False, encoding='utf-8-sig')

figure_records = []
for index in range(1, 8):
    for path in sorted((SN / 'Main').glob('Fig' + str(index) + '.*')):
        figure_records.append(dict(figure=index, path=str(path.relative_to(O)), bytes=path.stat().st_size, sha256=sha(path)))
pd.DataFrame(figure_records).to_csv(Q / '最终主图文件清单.csv', index=False, encoding='utf-8-sig')

source_zip = O / 'Source_Data_20261018.zip'
with ZipFile(source_zip, 'w', ZIP_DEFLATED, compresslevel=3) as output:
    selected = [I / '环境变量全量分类字典_20261018.csv', I / '环境指标分量与有效权重.csv']
    selected += sorted(p for p in D.glob('*.csv') if '新旧' not in p.name)
    selected += sorted((SN / 'Q2/02_source_data').glob('*.csv'))
    selected += sorted((SN / 'Q3/01_source_data').glob('*.csv'))
    selected += sorted((SN / 'Q4/02_source_data').glob('*.csv'))
    selected += sorted((D / '未受归类影响的背景数据').glob('*.csv'))
    for path in selected: output.write(path, str(path.relative_to(O)))
    output.writestr('README.txt', 'Population density is classified exclusively as direct human activity. Biological independent units: 86 river and 44 lake sites. Unchanged global GDP and descriptive water-quality context data are labelled separately. Historical comparison CSVs are audit records, not current figure inputs. Main figures use the per-figure source-data directories. Full scripts and frozen inputs accompany the local revision package.\n')
with ZipFile(source_zip) as archive: assert archive.testzip() is None

facts = json.loads((T / '正文数字证据表.json').read_text(encoding='utf8'))
lines = [
    '# 人口密度归类修订完成说明', '',
    '本次以2026年10月17日手稿为正文基线，将人口密度在生物固定指数和原始预测变量集中统一归入人类直接活动。城市建设仅保留建成地表与反向自然/绿色覆盖分量。原始观测和样本集合未改变。', '',
    '## 交付文件', '',
    '- Manuscript_20261018.docx：正文、Results、Discussion、摘要、结论及方法同步更新。',
    '- Supplementary_materials_20261018.docx：人口来源段落移至直接活动小节，统一两类响应系统和敏感性分析的分类定义。',
    '- 04_Figures/Main：按现稿Fig. 1–7编号整理；Fig. 5–7重绘，提供PNG、PDF、SVG、TIFF。Fig. 1–4保留原图，避免不必要的样式改变。',
    '- 04_Figures/Supplementary：原三张补充图；04_Figures/Context_unchanged保留未受归类影响的背景图。',
    '- 02_数据与统计、Source_Data_20261018.zip：重算结果及配套源数据。',
    '- 05_可复现脚本、01_输入与分类定义：脚本、原始输入快照、完整变量分类、权重与历史比较基线。',
    '- 06_QA：数值、分类、原生引文域、图件与Word排版核验记录；PDF和页面图仅为内部质控。', '',
    '## 重算范围', '',
    '1. 生物三类梯度、16,575条偏相关及分层FDR、390个响应维度与尺度的共同置换maxT检验。',
    '2. 178个河流、144个湖泊具体特征的12,558条尺度关联、999次置换、499次站点bootstrap、贡献与排名。',
    '3. 水质与生物的强度份额、内部秩轨迹、方向一致性及双端支持。',
    '4. 30个代表响应的1,170条空间阻塞交叉验证、群落矩阵pRDA及离散度、年份/水期模型、岭回归稳定性与河湖差异。',
    '5. 类群×五响应维度的2,145个尺度组合和配套图件。',
    '6. 水质梯度及117个结局×梯度×尺度组合独立复算，其中78个主结果的系数、置换P、FDR及区间与原值一致。',
    '7. 原环境子变量分解已把人口归入直接活动；重建输入逐值核对一致后保留其原结果。全球GDP和原始调查概况独立于此次归类变更。', '',
    '## 主要结果变化', '',
    '| 结果 | 原稿 | 本次结果 |',
    '|---|---|---|',
    '| 河流建设—群落组成 | 13个尺度支持，峰值2 km | 11个尺度支持，峰值2 km，RMS=0.240 |',
    '| 河流直接活动 | 功能/指示指标有支持 | 功能/指示指标9个尺度支持，群落组成在3.5 km亦有支持 |',
    '| 湖泊建设—β多样性 | 峰值1 km | 峰值1.5 km，RMS=0.288，maxT P=0.006 |',
    '| 湖泊直接活动 | 五维峰值分散在0.5–4 km | 五维仍均有支持，峰值0.5–2.5 km |',
    '| 河流活动前五特征 | 前三为鱼类体长 | 前两为鱼类体长，前五包含两项硅藻比例指标 |',
    '| 湖泊建设—底栖Sørensen LCBD | ρ=−0.570 | 1 km处ρ=−0.549，家族范围maxT P=0.025 |',
    '| 湖泊活动—总体丰富度 | ρ=0.564 | 1.5 km处ρ=0.582，家族范围maxT P=0.007 |', '',
    '完整数值见03_文本修订/正文数字证据表.json、逐段修订记录及前后比较CSV。正文结论仍围绕水质与生物的互补性、空间尺度依赖和河湖差异；未将关联改写为因果效应，也未按预期方向筛选结果。', '',
    '## 复现顺序', '',
    '使用ch5_config.py中的Python环境。依次运行ch5_prepare_analysis、ch5_primary_analysis、ch5_carrier_analysis、02_verify_water_quality、ch5_comparison_analysis、ch5_network_data、ch5_sensitivity_analysis、ch5_auxiliary_audit、08_rerun_taxon_family及12_plot_taxon_family。原版式绘图使用ch5_plot_revisions.py en Q2/Q3/Q4。最后运行05_build_evidence、06_prepare_prose、07_build_documents及核验脚本。', '',
    '原生引文和数学公式完整保留；正文含41处Zotero引文域，未改动引用内容及顺序。源Manuscript_20261017.docx、原始数据和学位论文文件不删除。旧版图件清理与恢复范围单列于07_旧版备份中的清单。',
]
(O / 'README_修订结果与复现说明.md').write_text('\n'.join(lines), encoding='utf8')
write_json(Q / '源数据打包核验.json', dict(passed=True, file=str(source_zip), sha256=sha(source_zip),
    figures=len(figure_records), source_csv_files=len(selected)))
print('Reproducibility snapshots, source data archive and revision summary prepared.', flush=True)
