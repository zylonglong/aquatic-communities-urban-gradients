import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from ch5_config import *
sys.stdout.reconfigure(encoding='utf8')
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
stages = sys.argv[1:] or ['ch5_prepare_analysis.py', 'ch5_primary_analysis.py', 'ch5_carrier_analysis.py']
for stage in stages:
    arguments = stage.split(':')
    print(datetime.now().isoformat(), 'START', stage, flush=True)
    log = Q / (Path(arguments[0]).stem + ('_' + '_'.join(arguments[1:]) if len(arguments) > 1 else '') + '.log')
    with log.open('w', encoding='utf8') as stream:
        result = subprocess.run([str(PY), '-u', str(S / arguments[0]), *arguments[1:]], cwd=S,
                                stdout=stream, stderr=subprocess.STDOUT, env=os.environ)
    print(datetime.now().isoformat(), 'EXIT', stage, result.returncode, flush=True)
    if result.returncode:
        print(log.read_text(encoding='utf8')[-12000:], flush=True)
        raise SystemExit(result.returncode)
