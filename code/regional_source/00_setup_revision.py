from ch5_config import *
import shutil
import sys
sys.stdout.reconfigure(encoding='utf8')

PREVIOUS = ROOT / '.1博士毕业材料/毕业论文数据分析/第五章_水生态系统对城市发展的多尺度响应与机制解析'
old_scripts = PREVIOUS / '04_可复现脚本'
old_inputs = PREVIOUS / '00_输入快照与指标定义'
names = ['ML_Input_Raw_13Ranges_Sample_Site.xlsx', '五类水生生物变量对应关系_Q2分析流程.xlsx',
         'Feature_Process_Aquatic_Biology_Raw.xlsx', 'Feature_Process_Beta_Diversity_Raw.xlsx',
         'Feature_Process_Biological_Response_Traits_Raw.xlsx']
manifest = []
for name in names:
    source = old_inputs / name
    destination = I / name
    shutil.copy2(source, destination)
    assert sha(source) == sha(destination)
    live = (SRC if name.startswith('五类') else ML) / name
    assert live.exists() and sha(live) == sha(source), f'Frozen/live input differs: {name}'
    manifest.append(dict(source=str(live), snapshot=str(destination), sha256=sha(destination)))
for source in [SCI, SUP]:
    destination = I / source.name
    shutil.copy2(source, destination)
    manifest.append(dict(source=str(source), snapshot=str(destination), sha256=sha(destination)))
for name in ['ch5_prepare_analysis.py', 'ch5_primary_analysis.py', 'ch5_carrier_analysis.py',
             'ch5_comparison_analysis.py', 'ch5_network_data.py', 'ch5_plot_revisions.py',
             'ch5_sensitivity_analysis.py', 'ch5_auxiliary_audit.py']:
    # Preserve this package's adapted scripts on repeated setup runs.
    if not (S / name).exists():
        shutil.copy2(old_scripts / name, S / name)
for name in ['原SCI脚本', '原绘图脚本']:
    if not (S / name).exists():
        shutil.copytree(old_scripts / name, S / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
write_json(I / '输入文件_SHA256.json', manifest)
write_json(O / '执行计划.json', dict(
    source_manuscript=str(SCI), independent_units=dict(River=86, Lake=44),
    events=dict(River=409, Lake=155), classification='population density exclusively in direct human activity',
    preserve=['responses', 'independent units', 'spatial controls', 'random seeds', 'permutation counts', 'plot layouts'],
    stages=['input and category audit', 'primary and carrier reanalysis', 'water-quality invariance audit',
            'spatial and temporal sensitivity analyses', 'cross-endpoint comparisons',
            'original-format figure regeneration', 'manuscript and supplement revision',
            'claim/evidence and rendered-page QA', 'verified recoverable cleanup'],
    protected=['raw observations', 'raw population rasters', 'thesis products', '20261017 source manuscript'],
))
print('Revision workspace initialized; frozen/live data hashes match.', flush=True)
