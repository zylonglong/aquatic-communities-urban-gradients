from ch5_config import *
import sys
import shutil
sys.stdout.reconfigure(encoding='utf8')
source = S / '原SCI脚本/07_taxon_family_cross_analysis.py'
text = source.read_text(encoding='utf-8-sig')
text = text.replace('from __future__ import annotations',
                    'from __future__ import annotations\nfrom ch5_config import *', 1)
text = text.replace('ml_root = out.parent / "ML_Input_原始量_13尺度_2026-07-04"', 'ml_root = I', 1)
text = text.replace('core = load_core(script_dir)',
                    'core = load_module("taxon_reclassified_core", S / "ch5_primary_core.py")', 1)
text = text.replace('out / "05_Q4_具体生物响应" / "全部具体响应_偏Spearman_分层FDR.csv"',
                    'D / "全部具体响应_偏Spearman_分层FDR.csv"')
text = text.replace('out / "05_Q4_具体生物响应" / "具体响应尺度类型与候选清单.csv"',
                    'D / "具体响应尺度类型与候选清单.csv"')
# The workbook digest duplicates the CSVs; keep the CSV deliverables authoritative.
start = text.index('    # Excel digest.')
end = text.index('    config = {', start)
text = text[:start] + text[end:]
target = S / 'taxon_family_reclassified.py'
target.write_text(text, encoding='utf8')
sub = O / '13_生物类群×五家族精细分析_2026-07-24'
for folder in ['00_配置与图件契约', '01_类群家族数据审计', '02_类群家族整体尺度检验',
               '03_类群内具体响应', '04_论文级图件', '05_SourceData', '06_QA']:
    (sub / folder).mkdir(parents=True, exist_ok=True)
module = load_module('taxon_family_reclassified', target)
module.main()
print('Taxon-by-domain supplementary analyses updated using the same reclassified core.', flush=True)
