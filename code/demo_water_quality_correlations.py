"""Recalculate regional water-quality Spearman coefficients from published inputs.

This entry point uses only Python's standard library and openpyxl. It does not
refit biological/global models or rerun permutation/interval estimation.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import argparse,json,math
from openpyxl import load_workbook

def records(workbook,sheet):
    rows=workbook[sheet].iter_rows(values_only=True)
    headers=next(rows)
    return [dict(zip(headers,row)) for row in rows if any(v is not None for v in row)]

def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    ranks=[0.0]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        value=(i+1+j)/2
        for k in range(i,j):ranks[order[k]]=value
        i=j
    return ranks

def spearman(x,y):
    a=rank(x);b=rank(y);ma=mean(a);mb=mean(b)
    numerator=sum((u-ma)*(v-mb) for u,v in zip(a,b))
    denominator=math.sqrt(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))
    return numerator/denominator if denominator else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('demo_results.json'))
    args=ap.parse_args()
    b=load_workbook(args.data_dir/'Supplementary_Data_1.xlsx',read_only=True,data_only=True)
    units={r['spatial_id']:r for r in records(b,'Water_quality_analysis_units')}
    gradients=records(b,'Water_quality_gradients');b.close()
    b=load_workbook(args.data_dir/'Supplementary_Data_4.xlsx',read_only=True,data_only=True)
    expected=records(b,'Fig3_water_quality');b.close()
    groups=defaultdict(list)
    for r in gradients:groups[(r['water_body_type'],r['driver'],int(r['range_m']))].append(r)
    checks=[]
    for e in expected:
        key=(e['water_body_type'],e['driver'],int(e['range_m']))
        pairs=[]
        for g in groups[key]:
            if g['spatial_id'] not in units:continue
            grade=units[g['spatial_id']]['grade'];score=g['gradient_score']
            if grade is not None and score is not None and math.isfinite(float(grade)) and math.isfinite(float(score)):
                pairs.append((float(score),float(grade)))
        rho=spearman([a for a,b in pairs],[b for a,b in pairs])
        delta=abs(rho-float(e['spearman_rho']))
        passed=len(pairs)==int(e['n_spatial_units']) and delta<1e-12
        checks.append({'water_body_type':key[0],'driver':key[1],'range_m':key[2],'n':len(pairs),'recomputed_rho':rho,'published_rho':e['spearman_rho'],'absolute_difference':delta,'passed':passed})
    result={'scope':'78 input-based Spearman coefficient recalculations; no resampling or full pipeline rerun','checks':len(checks),'passed':sum(c['passed'] for c in checks),'maximum_absolute_difference':max(c['absolute_difference'] for c in checks),'results':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
    if result['checks']!=78 or result['passed']!=78:raise SystemExit(1)

if __name__=='__main__':main()
