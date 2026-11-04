from ch5_config import *
import pandas as pd,sys,shutil
sys.stdout.reconfigure(encoding='utf8')
source=SRC/'04_Q3_水质与生物对照_响应强度重分析_2026-08-06/Q3_重分析脚本.py'
shutil.copy2(source,S/'ch5_q3_original.py')
m=load_module('ch5_q3_original',S/'ch5_q3_original.py')
wq=pd.read_csv(D/'水质等级_三类环境梯度_13尺度响应.csv');wq['environment_module']=wq.driver.map(m.DRIVER_TO_MODULE)
family=pd.read_csv(D/'环境模块_生态维度_13尺度全局置换检验.csv')
specific=pd.read_csv(D/'全部具体响应_偏Spearman_分层FDR.csv')
cells,family_cells=m.build_strength(wq,family)
body,body_pair,driver,indirect,pattern,trajectory=m.strength_summaries(cells)
dc,ds,priority=m.build_direction(wq,family_cells,cells)
ed,ee,es=m.build_ecological_direction(I/'五类水生生物变量对应关系_Q2分析流程.xlsx',specific,dc)
files={'Q3_强度分布_水质与生物总体逐range.csv':cells,'Q3_强度分布_五类生物逐range.csv':family_cells,'Q3_强度分布_水体内部汇总.csv':body,'Q3_强度分布_River_Lake配对描述.csv':body_pair,'Q3_强度分布_因素内部汇总.csv':driver,'Q3_强度分布_水质生物间接对照.csv':indirect,'Q3_强度分布_39单元结构对照.csv':pattern,'Q3_强度分布_range轨迹对照.csv':trajectory,'Q3_方向强度联合_五类逐range.csv':dc,'Q3_方向强度联合_梯度家族汇总.csv':ds,'Q3_方向强度联合_重点校正支持结果.csv':priority,'Q3_生态方向_指标字典.csv':ed,'Q3_生态方向_指标效应明细.csv':ee,'Q3_生态方向_梯度汇总.csv':es}
out=SN/'Q3/01_source_data';out.mkdir(parents=True,exist_ok=True)
for name,df in files.items():
 df.to_csv(D/name,index=False,encoding='utf-8-sig');df.to_csv(out/name,index=False,encoding='utf-8-sig')
assert len(cells)==156 and len(family)==390 and len(dc)==390
assert cells.groupby(['water_body_type','system']).size().eq(39).all()
write_json(Q/'双响应系统复算核对.json',dict(passed=True,cells=len(cells),family_direction_cells=len(dc),original_method_preserved=True,nested_ranges_treated_as_independent=False))
print('Q3 recomputed from updated biological responses and verified water-quality results',flush=True)
