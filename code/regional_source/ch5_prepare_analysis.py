from ch5_config import *
import sys,difflib
sys.stdout.reconfigure(encoding='utf8')
original=(S/'原SCI脚本/01_primary_multiscale_analysis.py').read_text(encoding='utf-8-sig')
new=original.replace('# Urbanization: positive built/population dimensions minus natural-buffer dimensions.','# Urban construction: positive built dimensions minus natural-buffer dimensions.')
needle='f"landuse_{tag}_urban_pct", f"worldpop_{tag}_mean_density",'
assert new.count(needle)==1
new=new.replace(needle,'f"landuse_{tag}_urban_pct",')
needle='hvars = [c for c in df.columns if c.startswith(f"api_{tag}_")]'
assert new.count(needle)==1
new=new.replace(needle,needle+'\n            hvars += [f"worldpop_{tag}_mean_density"]')
new=new.replace('"建设与人口压力"','"建设地表"')
new=new.replace('"不透水、建筑、建设用地、人口为正，自然用地与绿色植被为负，Robust-z后等权平均"','"不透水、建筑、建设用地为正，自然用地与绿色植被为负；人口已移至直接人类活动；Robust-z等权平均"')
new=new.replace('"P1–P8点密度与面占用先log1p再Robust-z并等权平均"','"P1–P8点密度、面占用及人口密度先log1p再Robust-z并等权平均"')
(S/'ch5_primary_core.py').write_text(new,encoding='utf8')
(Q/'人口归属调整_主分析代码差异.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='original_core',tofile='population_reclassified_core')),encoding='utf8')
cv=(S/'原SCI脚本/02_spatial_cv_and_mechanisms.py').read_text(encoding='utf-8-sig')
old='        p = f"worldpop_{t}_mean_density"\n        if p in df: cols.append(p)\n    else:\n        cols = [c for c in df.columns if c.startswith(f"api_{t}_")]'
newcv='    else:\n        cols = [c for c in df.columns if c.startswith(f"api_{t}_")]\n        p = f"worldpop_{t}_mean_density"\n        if p in df: cols.append(p)'
assert old in cv
cv=cv.replace(old,newcv)
cv=cv.replace('np.fill_diagonal(corr.values, 1)\n    dist = 1 - corr.values','corr_array = corr.to_numpy(copy=True)\n    np.fill_diagonal(corr_array, 1)\n    dist = 1 - corr_array')
(S/'ch5_cv_core.py').write_text(cv,encoding='utf8')
# Imports of the original water-quality script must only create output folders
# under the new revision workspace; none of its original folders are touched.
wq=(WQ/'05_脚本/02_build_wq_input_and_analyze.py').read_text(encoding='utf-8-sig')
wq=wq.replace('OUT = Path(__file__).resolve().parent.parent','OUT = Path(r"'+str(D/'Q1_水质重算')+'")')
(S/'ch5_water_core.py').write_text(wq,encoding='utf8')
write_json(I/'人口密度指标移类契约.json',dict(original_city_components=['impervious_pct','building_pct','urban_pct','population_density','negative natural_land_pct','negative green_pct'],new_city_components=['impervious_pct','building_pct','urban_pct','negative natural_land_pct','negative green_pct'],new_activity_components='原P1至P8点密度及面覆盖分量，加同尺度人口密度分量；同原活动流程log1p、robust-z、有效分量等权平均、组合后robust-z',water_quality_definition='原水质PCA已将人口归入直接活动，城市建设候选集不含人口；复算以验证不变',spatial_cv_definition='城市建设原始变量删除人口；直接活动原始变量加入人口',unchanged=['冻结生物响应变量与样本','水质观测','地形地貌指标','空间控制','随机种子及重采样次数','原图版式'],sci_document_edits=False))
print('Scoped reclassification cores prepared; original scripts preserved.')
