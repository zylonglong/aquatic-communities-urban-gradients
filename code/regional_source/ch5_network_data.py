from ch5_config import *
import shutil,sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf8')
helpers=S/'绘图适配';helpers.mkdir(exist_ok=True)
names=['q4_hierarchical_response_network.py','q4_lake_semicircular_smooth_tracks.py','q4_river_lake_shared_semicircular_network.py','fig5b_top5_driver_responses.py']
for name in names:shutil.copy2(S/'冻结绘图依赖'/name,helpers/name)
sys.path.insert(0,str(helpers))
net=load_module('network_source_core',helpers/names[0])
net.SOURCE=SN/'Q4/02_source_data';net.Q2_FAMILY=D/'环境模块_生态维度_13尺度全局置换检验.csv'
variable,family=net.prepare_metrics()
nd=SN/'Q4/05_network_redesign/02_source_data';nd.mkdir(parents=True,exist_ok=True)
variable.to_csv(nd/'Q4_network_variable_metrics.csv',index=False,encoding='utf-8-sig');family.to_csv(nd/'Q4_network_dimension_metrics.csv',index=False,encoding='utf-8-sig')
import q4_lake_semicircular_smooth_tracks as base
base.BASE=SN/'Q4/05_network_redesign';base.NETWORK_DATA=nd
shared=SN/'Q4/05_network_redesign/09_river_lake_shared/02_source_data';shared.mkdir(parents=True,exist_ok=True)
for w in ['River','Lake']:
 ordered,_=base.build_ordered_data(w)
 ordered.to_csv(shared/('Q4_'+w.lower()+'_ordered_variable_metrics.csv'),index=False,encoding='utf-8-sig')
import fig5b_top5_driver_responses as b
b.NETWORK_SOURCE=shared;b.ASSOCIATION_SOURCE=SN/'Q4/02_source_data/Q4_all_frozen_candidate_multiscale_associations.csv'
selected,boundary=b.load_and_select()
dest=SN/'Q4/06_panel_b_top5_driver_responses/02_source_data';dest.mkdir(parents=True,exist_ok=True)
selected.to_csv(dest/'Fig5b_top5_driver_responsive_characteristics.csv',index=False,encoding='utf-8-sig')
selected.to_csv(SN/'Q4/02_source_data/Fig5_panel_b_top5_driver_responsive_characteristics.csv',index=False,encoding='utf-8-sig')
selected.to_csv(D/'Q4_各水体各环境梯度前5响应特征.csv',index=False,encoding='utf-8-sig')
boundary.to_csv(Q/'Q4_前5排名边界.csv',index=False,encoding='utf-8-sig')
for frame,name in [(variable,'Q4_网络全变量指标.csv'),(family,'Q4_网络响应维度指标.csv')]:frame.to_csv(D/name,index=False,encoding='utf-8-sig')
print('Network metrics and top-five panels updated; '+str(len(selected))+' top-five entries.',flush=True)
