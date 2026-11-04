from ch5_config import *
import pandas as pd,numpy as np,sys,time
sys.stdout.reconfigure(encoding='utf8')
core=load_module('ch5_primary_core',S/'ch5_primary_core.py')
oldcore=load_module('ch5_original_core',S/'原SCI脚本/01_primary_multiscale_analysis.py')
print('Reading frozen site and event inputs',flush=True)
site=pd.read_excel(I/'ML_Input_Raw_13Ranges_Sample_Site.xlsx',sheet_name='Site_Special')
sample=pd.read_excel(I/'ML_Input_Raw_13Ranges_Sample_Site.xlsx',sheet_name='Sample_Special')
site.to_pickle(D/'冻结Site输入.pkl');sample.to_pickle(D/'冻结Sample输入.pkl')
meta=core.read_dictionaries(I);availability=core.response_availability(site,meta)
meta.to_csv(I/'冻结响应变量字典.csv',index=False,encoding='utf-8-sig')
availability.to_csv(I/'冻结响应可用性.csv',index=False,encoding='utf-8-sig')
original_scores,om,oc=oldcore.build_module_scores(site)
scores,sm,sc=core.build_module_scores(site)
frozen_scores=pd.read_csv(I/'历史基线_仅供差值核验/原城市建设含人口_固定指数.csv')
keys=['row_index','water_body_type','range_m','environment_module']
cmp=original_scores.merge(frozen_scores,on=keys,suffixes=('_reproduced','_frozen'),validate='1:1')
maxerr=float(np.max(np.abs(cmp.environment_score_reproduced-cmp.environment_score_frozen)))
assert len(cmp)==len(original_scores) and maxerr<1e-12
delta=scores.merge(original_scores,on=keys,suffixes=('_new','_old'),validate='1:1')
delta['score_change']=delta.environment_score_new-delta.environment_score_old
assert delta[delta.environment_module=='terrain'].score_change.abs().max()<1e-14
assert not sc[(sc.environment_module=='urbanization')&sc.variable.str.contains('worldpop')].shape[0]
assert sc[(sc.environment_module=='human_activity')&sc.variable.str.contains('worldpop')].shape[0]==26
scores.to_csv(D/'环境模块13尺度固定指数_人口移类.csv',index=False,encoding='utf-8-sig')
sm.to_csv(I/'环境模块固定指数构建清单_人口移类.csv',index=False,encoding='utf-8-sig')
sc.to_csv(I/'环境指数分量与方向权重_人口移类.csv',index=False,encoding='utf-8-sig')
delta.to_csv(D/'环境指数新旧逐点位比较.csv',index=False,encoding='utf-8-sig')
write_json(Q/'原指数基线复现.json',dict(rows=len(cmp),max_error=maxerr,passed=True,source_site_rows=len(site),source_event_rows=len(sample),site_counts=site.groupby('water_body_type').size().to_dict(),event_counts=sample.groupby('water_body_type').size().to_dict()))
print('Score baseline reproduced; computing biological associations and 499 common permutations',flush=True)
assoc,family,red=core.compute_associations(site,meta,availability,scores)
for df,name in [(assoc,'全部具体响应_偏Spearman_分层FDR.csv'),(family,'环境模块_生态维度_13尺度全局置换检验.csv'),(red,'家族全局检验响应冗余控制映射.csv')]:df.to_csv(D/name,index=False,encoding='utf-8-sig')
candidates=core.summarize_candidates(assoc,family);candidates.to_csv(D/'具体响应尺度类型与候选清单.csv',index=False,encoding='utf-8-sig')
oldfam=pd.read_csv(I/'历史基线_仅供差值核验/原城市建设含人口_响应维度检验.csv')
fk=['water_body_type','environment_module','response_family','range_m']
fc=family.merge(oldfam,on=fk,suffixes=('_new','_old'),validate='1:1')
tc=fc[fc.environment_module=='terrain'];err={c:float((tc[c+'_new']-tc[c+'_old']).abs().max()) for c in ['rms_partial_rho','median_partial_rho','range_maxT_p']}
assert max(err.values())<1e-12,err
fc.to_csv(D/'群落响应维度新旧比较.csv',index=False,encoding='utf-8-sig')
write_json(Q/'主分析复算核对.json',dict(passed=True,association_rows=len(assoc),family_rows=len(family),terrain_unchanged_max_errors=err,population_in_urban=False,population_in_activity=True))
for folder in [SN/'Q2/02_source_data',SN/'Q2/03_figure',SN/'Q2/04_QA']:folder.mkdir(parents=True,exist_ok=True)
family.to_csv(SN/'Q2/02_source_data/Figure03_Q2_biology_family_scale.csv',index=False,encoding='utf-8-sig')
print('Primary biological reanalysis complete',flush=True)
