from ch5_config import *
import pandas as pd,numpy as np,sys,shutil
sys.stdout.reconfigure(encoding='utf8')
core=load_module('carrier_modified_core',S/'ch5_primary_core.py')
q4=load_module('carrier_source',S/'原绘图脚本/Q4/01_script/q4_specific_biological_carriers.py')
site=pd.read_pickle(D/'冻结Site输入.pkl')
mapping=pd.read_excel(I/'五类水生生物变量对应关系_Q2分析流程.xlsx',sheet_name='Q2最终代表变量')
reps=q4.prepare_metadata(mapping);scores=pd.read_csv(D/'环境模块13尺度固定指数_人口移类.csv')
groups=[[],[],[],[],[],[]]
for w in q4.WATERS:
 print('Computing '+w+' response carriers: 499 site bootstraps and 999 common permutations',flush=True)
 result=q4.calculate_water(core,site,reps,scores,w)
 for group,df in zip(groups,result):group.append(df)
assoc,contrib,cmod,fshare,tshare,duplicates=[pd.concat(group,ignore_index=True) for group in groups]
selected=contrib[contrib.observed_rank_within_family<=3].copy()
files={'Q4_all_frozen_candidate_multiscale_associations.csv':assoc,'Q4_variable_contribution_and_bootstrap_stability.csv':contrib,'Q4_module_specific_variable_contributions.csv':cmod,'Q4_selected_top3_variables_per_family.csv':selected,'Figure_Q4_selected_variable_multiscale_source_data.csv':assoc[assoc.selected_top3],'Figure_Q4_family_signal_shares.csv':fshare,'Q4_taxon_signal_shares_within_family.csv':tshare,'Q4_rebuilt_environment_module_scores.csv':scores,'Q4_exact_duplicate_representative_audit.csv':duplicates}
dest=SN/'Q4/02_source_data';dest.mkdir(parents=True,exist_ok=True)
for name,df in files.items():
 df.to_csv(D/name,index=False,encoding='utf-8-sig');df.to_csv(dest/name,index=False,encoding='utf-8-sig')
old=pd.read_csv(I/'历史基线_仅供差值核验/原城市建设含人口_Q4尺度关联.csv')
keys=['water_body_type','response_family','column_name','environment_module','range_m']
cmp=assoc.merge(old,on=keys,suffixes=('_new','_old'),validate='1:1')
terrain=cmp[cmp.environment_module=='terrain']
err={c:float((terrain[c+'_new']-terrain[c+'_old']).abs().max()) for c in ['partial_spearman_rho','response_13range_maxT_p','family_variables_13range_maxT_p']}
assert len(cmp)==len(assoc)==len(old) and max(err.values())<1e-12,err
cmp.to_csv(D/'Q4_原变量尺度关联新旧比较.csv',index=False,encoding='utf-8-sig')
write_json(Q/'Q4_复算核对.json',dict(passed=True,variables=contrib.groupby('water_body_type').size().to_dict(),association_rows=len(assoc),bootstraps=q4.N_BOOT,permutations=q4.N_PERM,terrain_unchanged_errors=err))
print('Carrier reanalysis complete',flush=True)
