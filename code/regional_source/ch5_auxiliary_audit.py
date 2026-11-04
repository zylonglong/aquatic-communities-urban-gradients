from ch5_config import *
import pandas as pd,numpy as np,sys,shutil
sys.stdout.reconfigure(encoding='utf8')
site=pd.read_pickle(D/'冻结Site输入.pkl');scores=pd.read_csv(D/'环境模块13尺度固定指数_人口移类.csv')
core=load_module('aux_core',S/'ch5_primary_core.py')
cv=load_module('aux_cv',S/'ch5_cv_core.py')
assoc=pd.read_csv(D/'全部具体响应_偏Spearman_分层FDR.csv')
diff=cv.river_lake_effect_differences(assoc)
diff.to_csv(D/'River_Lake标准化效应差_FisherZ检验.csv',index=False,encoding='utf-8-sig')
sub=load_module('subvar_original',S/'原SCI脚本/13_environment_subvariable_decomposition.py')
assert any(x[0]=='population_density' for x in sub.SUBVARIABLES['human_activity'])
assert not any(x[0]=='population_density' for x in sub.SUBVARIABLES['urbanization'])
oldpath=SRC/'14_三类环境子变量×生物五家族尺度分解_2026-07-27'
urban=pd.read_excel(ML/'Feature_Process_Urbanization_OSM_13Ranges.xlsx',sheet_name='Process_Long')
new,audit=sub.build_subvariable_long(site,urban)
old=pd.read_csv(oldpath/'01_环境子变量审计/三类环境子变量_130站点_13尺度长表.csv')
keys=['row_index','site','water_body_type','environment_module','environment_subvariable','range_m']
compare=new.merge(old,on=keys,suffixes=('_recomputed','_archived'),validate='1:1')
err=float((compare.value_recomputed-compare.value_archived).abs().max());assert len(compare)==len(new)==len(old) and err<1e-10,err
new.to_csv(D/'环境子变量_原正确人口归属_独立复算.csv',index=False,encoding='utf-8-sig')
api=sub.build_weighted_api(site);api.to_csv(D/'API专家加权_原13尺度构建.csv',index=False,encoding='utf-8-sig')
dest=D/'子变量分解_原归属正确_复核保留';dest.mkdir(exist_ok=True)
manifest=[]
for p in oldpath.rglob('*.csv'):
 if any(x in p.parts for x in ['workbook_previews','__pycache__']):continue
 t=dest/p.relative_to(oldpath);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
 manifest.append(dict(source=str(p),copy=str(t),sha256=sha(p),unchanged=True))
write_json(Q/'子变量分解及水体差异复核.json',dict(passed=True,population_already_in_activity_in_original_subvariable_analysis=True,rows=len(compare),max_error=err,independent_group_assignment_verified=True,river_lake_effect_difference_rows=len(diff),preserved_subvariable_outputs=manifest))
print('Subvariable construction independently reproduced; original correct population membership retained. River/Lake differences updated.',flush=True)
