from pathlib import Path
import importlib.util
import json
import hashlib
import mimetypes

# MIME discovery does not need access to the Windows registry in this workflow.
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None

ROOT = Path(r'E:\桌面\博士课题\Data')
O = ROOT / 'manuscript/人口密度归类修订_20261018'
SRC = ROOT / 'manuscript/城市水生态四科学问题_尺度响应分析_2026-07-23'
SN = O / '04_Figures'
I = O / '01_输入与分类定义'
D = O / '02_数据与统计'
F = O / '04_Figures/中文备用'
T = O / '03_文本修订'
S = O / '05_可复现脚本'
Q = O / '06_QA'
for p in [I, D, F, T, S, Q, SN]:
    p.mkdir(parents=True, exist_ok=True)
ML = ROOT / 'data/processed/训练数据汇总/ML_Input_原始量_13尺度_2026-07-04'
WQ = ROOT / 'data/processed/WQ_13尺度环境特征与水质响应分析_2026-07-20'
GDP = SRC / '19_全球淡水经济发展与水生态宏观分析_更新版_2026-09-14/11_近年GDP更新_5x5布局_2026-09-15'
SCI = ROOT / 'manuscript/Manuscript_20261017.docx'
SUP = ROOT / 'manuscript/Supplementary materials.docx'
PY = ROOT / '.codex_envs/urban_aquatic_sci/Scripts/python.exe'
DOCPY = Path(r'C:\Users\guan9\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
NODE = Path(r'C:\Users\guan9\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
