from ch5_config import *
import pandas as pd,numpy as np,sys,time
sys.stdout.reconfigure(encoding='utf8')
cvcore=load_module('ch5_cv_core',S/'ch5_cv_core.py')
com=load_module('ch5_community_core',S/'原SCI脚本/03_community_and_temporal.py')
site=pd.read_pickle(D/'冻结Site输入.pkl');sample=pd.read_pickle(D/'冻结Sample输入.pkl')
scores=pd.read_csv(D/'环境模块13尺度固定指数_人口移类.csv')
candidates=pd.read_csv(D/'具体响应尺度类型与候选清单.csv');redundancy=pd.read_csv(D/'家族全局检验响应冗余控制映射.csv')
targets=cvcore.select_cv_targets(candidates,redundancy)
targets.to_csv(D/'空间诊断预设代表响应清单.csv',index=False,encoding='utf-8-sig')
cp=D/'代表响应_重复空间阻塞CV_13尺度.csv'
rows=[]
if cp.exists():rows=pd.read_csv(cp).to_dict('records')
done={(r['water_body_type'],r['environment_module'],r['response_variable']) for r in rows}
for i,t in enumerate(targets.itertuples()):
 key=(t.water_body_type,t.environment_module,t.column_name)
 if key in done:continue
 print('Spatial CV target '+str(i+1)+'/'+str(len(targets))+' '+str(key),flush=True)
 d=site[site.water_body_type==t.water_body_type].copy();y,rule=cvcore.response_transform(d[t.column_name],t.column_name,t.response_family)
 for r in RANGES:
  cols=cvcore.raw_module_variables(d,t.environment_module,r)
  if t.environment_module=='urbanization':assert not any('worldpop' in c for c in cols)
  if t.environment_module=='human_activity':assert any('worldpop' in c for c in cols)
  for seed in cvcore.SEEDS:
   result=cvcore.spatial_cv_one(d,np.asarray(y,float),cols,t.water_body_type,seed)
   if result:rows.append(dict(water_body_type=t.water_body_type,environment_module=t.environment_module,response_family=t.response_family,response_variable=t.column_name,response_name_cn=t.variable_name_cn,range_m=r,n_module_predictors=len(cols),response_transform=rule,**result))
 pd.DataFrame(rows).to_csv(cp,index=False,encoding='utf-8-sig')
 print('Spatial CV target saved',flush=True)
cv=pd.DataFrame(rows)
summary=cv.groupby(['water_body_type','environment_module','response_family','response_variable','range_m']).agg(n_repeats=('seed','nunique'),n_sites=('n_sites','median'),delta_cv_r2_median=('delta_cv_r2','median'),delta_cv_r2_min=('delta_cv_r2','min'),delta_cv_r2_max=('delta_cv_r2','max'),positive_delta_r2_fraction=('delta_cv_r2',lambda x:np.mean(np.asarray(x)>0))).reset_index()
summary.to_csv(D/'代表响应_空间CV汇总.csv',index=False,encoding='utf-8-sig')
print('Community-matrix pRDA and dispersion sensitivity',flush=True)
scores['site']=[site.loc[int(i),'site'] for i in scores.row_index]
rp=D/'原始分类单元矩阵_Hellinger偏RDA与群落离散度.csv'
if not rp.exists():
 result,taxa,ord= com.community_analysis(O,ML,site,scores)
 for df,name in [(result,rp.name),(taxa,'峰值尺度_分类单元偏相关与FDR.csv'),(ord,'群落组成排序坐标_峰值尺度.csv')]:df.to_csv(D/name,index=False,encoding='utf-8-sig')
print('Year/flow-period mixed-model sensitivity',flush=True)
tp=D/'Sample层年份水文期混合模型稳健性.csv'
if tp.exists():temp=pd.read_csv(tp)
else:
 temp=com.temporal_analysis(O,ML,site,sample,scores,targets)
 temp.to_csv(tp,index=False,encoding='utf-8-sig')
env_names={}
for fn in ['Feature_Process_DEM_Terrain_13Ranges.xlsx','Feature_Process_Urbanization_OSM_13Ranges.xlsx','Feature_Process_WorldCover_13Ranges.xlsx','Feature_Process_WorldPop_13Ranges.xlsx','Feature_Process_API_Raw_13Ranges.xlsx']:
 x=pd.read_excel(ML/fn,sheet_name='Feature_Dictionary');env_names.update(dict(zip(x.column_name,x.variable_name_cn)))
print('Within-module predictor and Ridge bootstrap sensitivity',flush=True)
single,boot,clusters=cvcore.mechanism_stability(site,targets,env_names)
for df,name in [(single,'代表响应_模块内具体变量跨尺度偏相关.csv'),(boot,'峰值尺度_共线性控制后Ridge_Bootstrap稳定性.csv'),(clusters,'模块内预测变量相关聚类映射.csv')]:df.to_csv(D/name,index=False,encoding='utf-8-sig')
write_json(Q/'敏感性重算完成.json',dict(passed=True,targets=len(targets),cv_rows=len(cv),mixed_terms=len(temp),population_reclassified_in_raw_predictor_sets=True))
print('Sensitivity analyses complete',flush=True)
