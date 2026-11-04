from ch5_config import *
import sys
sys.stdout.reconfigure(encoding='utf8')
source = S / '原SCI脚本/08_taxon_family_publication_figures.py'
text = source.read_text(encoding='utf-8-sig')
text = text.replace('from __future__ import annotations',
                    'from __future__ import annotations\nfrom ch5_config import *', 1)
text = text.replace('"human_activity": "人类活动压力"', '"human_activity": "人类直接活动"')
target = S / 'taxon_family_plots_reclassified.py'
target.write_text(text, encoding='utf8')
sub = O / '13_生物类群×五家族精细分析_2026-07-24'
for folder in ['PNG', 'PDF', 'SVG', 'TIFF', '类群分面PNG', '类群分面PDF']:
    (sub / '04_论文级图件' / folder).mkdir(parents=True, exist_ok=True)
module = load_module('taxon_family_plots_reclassified', target)
module.main()
