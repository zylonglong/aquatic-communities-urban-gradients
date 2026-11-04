from ch5_config import *
import sys,shutil,importlib,re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
sys.stdout.reconfigure(encoding='utf8')
LANG=sys.argv[1] if len(sys.argv)>1 else 'en';PART=sys.argv[2] if len(sys.argv)>2 else 'all'
ZH=LANG=='zh'
ACTIVE_TRANSLATION=ZH
exact={'Topography and geomorphology':'地形地貌','Topography and\ngeomorphology':'地形地貌','Urban construction':'城市建设','Urban\nconstruction':'城市建设','Direct human activity':'直接人类活动','Direct human\nactivity':'直接人类活动','Direct activity':'直接活动','community-level attributes':'群落基本属性','community-level\nattributes':'群落基本属性','Community-level attributes':'群落基本属性','Community-level\nattributes':'群落基本属性','α diversity':'α多样性','β diversity':'β多样性','community composition':'群落组成','community\ncomposition':'群落组成','Community composition':'群落组成','Community\ncomposition':'群落组成','indicator/tolerance taxa':'功能与指示特征','indicator/tolerance\ntaxa':'功能与指示特征','Functional or indicator metrics':'功能与指示特征','Functional or\nindicator\nmetrics':'功能与指示特征','Functional/indicator':'功能与指示特征','River':'河流','Lake':'湖库','Water quality':'水质','Aquatic biota':'水生生物','Aquatic biology':'水生生物','Spatial range (km)':'空间范围（km）','Scale-dependent associations':'多尺度关联','Statistical evidence across scales':'各空间范围的统计证据','Scale-dependent response domains':'多尺度群落响应','Integrated response strength across domains':'跨响应维度的整体强度','Overall RMS strength':'整体RMS强度','Response strength (RMS partial ρ)':'响应强度（RMS偏ρ）','Range-wise maxT P ≤ 0.05':'范围校正maxT P ≤ 0.05','Negative':'负向','Positive':'正向','Signed median partial Spearman ρ':'偏Spearman ρ中位数','Spearman ρ':'Spearman ρ','95% CI':'95%置信区间','Response-domain min–max':'响应维度最小至最大值','Permutation P ≤ 0.05':'置换P ≤ 0.05','FDR q ≤ 0.05':'FDR q ≤ 0.05','Permutation P (darker = smaller)':'置换P（越深越小）','FDR q — bubble size (smaller = larger)':'FDR q映射点面积（越小越大）','Water-quality and aquatic-biota responses':'水质与水生生物响应','Waterbody':'水体类型','Within-system response strength':'系统内部响应强度','Within-system response distributions':'系统内部响应分布','Proportional allocation of response strength':'响应强度的比例分配','Paired internal response ranks':'配对内部响应秩','Directional and strength concordance':'方向与强度一致性','Ecological response domains':'生态响应维度','Biological response characteristics':'生物响应特征','Characteristic rank':'生物特征排序','RMS response strength':'RMS响应强度','Cross-gradient RMS strength':'跨环境梯度RMS强度','Balanced contribution':'均衡贡献','Strict family-level maxT':'严格家族maxT','Response-specific maxT':'单指标maxT'}
phrases={'River (n = 19)':'河流（n = 19）','Lake (n = 35)':'湖库（n = 35）','River (n = 86)':'河流（n = 86）','Lake (n = 44)':'湖库（n = 44）','water quality':'水质','aquatic biota':'水生生物','waterbody':'水体类型','Water quality':'水质','Aquatic biota':'水生生物','Topography and geomorphology':'地形地貌','Urban construction':'城市建设','Direct human activity':'直接人类活动','Response strength':'响应强度','response strength':'响应强度','signed median':'带方向中位数','Internal strength rank':'内部强度秩','Percentile':'百分位','Direction agreement':'方向一致性','Strength-trajectory agreement':'强度轨迹一致性','Weighted direction agreement':'加权方向一致性','Mean joint percentile strength':'平均联合百分位强度','Joint percentile strength':'联合百分位强度','Response-specific':'单指标','family-level':'家族','Supported':'有校正支持','supported':'有校正支持','not supported':'无校正支持','No support':'无校正支持','Positive':'正向','Negative':'负向','Recent economic context':'近期经济发展背景','Aquatic-biota observations':'水生生物观测','Adjusted GDP effects':'调整后的经济水平关联','Biotic responses':'水生生物响应','Water-quality observations':'水质观测','Water-quality responses':'水质响应','Adjusted response (SD)':'调整后响应（SD）','Standardized GDP effect (95% CI)':'标准化GDP系数（95%置信区间）','GDPpc distribution':'GDPpc分布','GDPpc\ndistribution':'GDPpc\n分布','Fish richness':'鱼类丰富度','EPTO richness':'EPTO丰富度','Chlorophyll a':'叶绿素a','DO saturation':'溶解氧饱和度'}
exact.update({'Ecological dimensions carrying the overall biological signal':'承担整体生物响应的生态维度','Strongest biological responses within environmental drivers':'各环境梯度下的主要生物响应','Multiscale responses across all biological characteristics':'全部生物特征的多尺度响应','Ecological response dimensions':'生态响应维度','Signed median ρ':'ρ中位数（带方向）','Statistical support':'统计支持','strict family maxT':'维度内严格maxT','response-specific 13-range maxT only':'仅单指标13范围maxT','Variable → response domain':'生物特征至响应维度','Driver → response domain':'环境梯度至响应维度','Line style: family maxT support':'线型：响应维度maxT支持','Colour: overall signed median ρ':'颜色：整体ρ中位数（带方向）','13-range RMS response strength':'13范围RMS响应强度','Area height = 13-range RMS response strength; fill colour = variable-specific signed median ρ.':'面积高度：13范围RMS强度；填色：各指标ρ中位数（带方向）','Line width = balanced contribution across the three environmental gradients (equal gradient weights).':'线宽：三类环境梯度等权的均衡贡献','Width = response-domain RMS across all variables and 13 ranges.':'箭头宽度：该维度所有指标及13范围的RMS强度','Line/point colour: 13-range signed median ρ':'线与点颜色：13范围ρ中位数（带方向）','Within-system response-strength distributions':'系统内部响应强度分布','Allocation of response strength within each system':'各系统内部响应强度的分配','Paired water-quality and aquatic-biota strength ranks':'水质与生物内部强度秩的配对','Family-specific direction and strength concordance':'各响应维度的方向与强度一致性','Across waterbodies':'水体类型之间','Across drivers within each waterbody and response system':'各水体与响应系统内部的环境梯度分配','Environmental driver':'环境梯度','Strength percentile (within system)':'系统内部强度百分位','Within-system strength rank (1 = strongest)':'系统内部强度秩（1为最强）','Water quality: |ρ| rank':'水质：|ρ|的秩','Aquatic biota: mean family RMS rank':'水生生物：五维RMS均值的秩','Aquatic-biota rank':'水生生物的秩','Water-quality rank':'水质的秩','Same driver × range pair':'相同梯度与范围配对','Dashed: ρ < 0':'虚线：ρ < 0','Mean joint percentile strength (point area; %)':'平均联合强度百分位（点面积；%）','Weighted direction agreement':'加权方向一致性','Same sign':'同向','Mixed':'混合','Opposite':'反向','≥1 dual-supported range':'至少1个范围有双端支持','no supported range':'无校正支持范围','≥1 supported range':'至少1个范围有校正支持','positive':'正向','negative':'负向','weak or mixed':'弱或混合','weak/mixed':'弱或混合','low':'低','medium':'中','high':'高'})
phrases.update({'River':'河流','Lake':'湖库','Aquatic-biota':'水生生物','α diversity':'α多样性','β diversity':'β多样性','Mean joint percentile strength':'平均联合强度百分位','(point area; %)':'（点面积；%）','Strength percentile':'强度百分位','(within system)':'（系统内部）','Same driver × range':'相同梯度与范围','Water quality: |ρ| rank':'水质：|ρ|的秩','mean family RMS rank':'五维RMS均值的秩','rank':'秩','Median |ρ|':'中位数|ρ|','dual-supported':'双端支持','range':'范围','Across waterbodies':'水体类型之间'})
exact.update({r'$\alpha$ diversity':'α多样性',r'$\beta$ diversity':'β多样性','Mean joint percentile strength (point area; %)':'联合强度百分位（点面积；%）'})
phrases.update({'Median':'中位数','Dashed:':'虚线：',r'$\alpha$ diversity':'α多样性',r'$\beta$ diversity':'β多样性'})
# Canonical wording follows the user's category names.
exact={k:v.replace('直接人类活动','人类直接活动') for k,v in exact.items()};phrases={k:v.replace('直接人类活动','人类直接活动') for k,v in phrases.items()}
if ZH:
 old_text=Text.set_text
 def chinese_text(self,text):
  if ACTIVE_TRANSLATION and isinstance(text,str):
   text=exact.get(text,exact.get(re.sub(r'\s+',' ',text).strip(),text))
   for a,b in phrases.items():text=text.replace(a,b)
   if hasattr(self,'_fontproperties') and any('\u4e00'<=ch<='\u9fff' for ch in text):self.set_fontfamily(['Microsoft YaHei','Arial'])
  return old_text(self,text)
 Text.set_text=chinese_text
 old_layout=Text._get_layout
 def chinese_layout(self,renderer):
  if any('\u4e00'<=ch<='\u9fff' for ch in self.get_text()) or self.get_text() in list('abcdef'):self.set_fontfamily(['Microsoft YaHei','Arial'])
  if self.get_text() in list('abcdef'):self.set_fontweight('bold')
  return old_layout(self,renderer)
 Text._get_layout=chinese_layout

def make_module(name,path):
 mod=load_module(name,path);return mod
def finish_zh(root,kind,stem):
 files=[]
 for p in root.rglob('*'):
  if p.is_file() and p.suffix.lower() in ['.png','.svg','.pdf','.tiff'] and ('03_figure' in p.parts):
   target=F/(stem+p.suffix.lower());shutil.copy2(p,target);files.append(str(target))
 write_json(Q/(kind+'_中文图件输出.json'),files)

if PART in ['all','Q1','Q2']:
 for kind,stem,file in [('Q1','图46_水质对三类环境梯度的多尺度响应','plot_q1_water_quality_multiscale.py'),('Q2','图47_水生生物群落五类响应维度的多尺度特征','plot_q2_biology_response_domains_bubbles.py')]:
  if PART not in ['all',kind]:continue
  out=SN/kind if not ZH else Q/'中文重绘'/kind
  for folder in ['01_script','02_source_data','03_figure','04_QA']:(out/folder).mkdir(parents=True,exist_ok=True)
  data=SN/kind/'02_source_data'
  for p in data.glob('*.csv'):shutil.copy2(p,out/'02_source_data'/p.name) if ZH else None
  script=next((S/'原绘图脚本'/kind).rglob(file));target=out/'01_script'/file;shutil.copy2(script,target)
  m=make_module('plot_'+kind+'_'+LANG,target)
  m.main()
  if ZH:finish_zh(out,kind,stem)
  print(kind+' '+LANG+' complete',flush=True)

if PART in ['all','Q3']:
 out=SN/'Q3' if not ZH else Q/'中文重绘/Q3'
 for f in ['01_source_data','02_script','03_figure','04_QA']:(out/f).mkdir(parents=True,exist_ok=True)
 if ZH:
  for p in (SN/'Q3/01_source_data').glob('*.csv'):shutil.copy2(p,out/'01_source_data'/p.name)
 src=next((S/'原绘图脚本/Q3').rglob('plot_q3_internal_distribution_direction.py'));target=out/'02_script'/src.name;shutil.copy2(src,target)
 m=make_module('plot_Q3_'+LANG,target);m.make_figure(out/'01_source_data',out/'03_figure')
 if ZH:finish_zh(out,'Q3','图49_水质与水生生物响应的一致性与分异')
 print('Q3 '+LANG+' complete',flush=True)

if PART in ['all','Q4']:
 helpers=S/'绘图适配';sys.path.insert(0,str(helpers))
 import q4_lake_semicircular_smooth_tracks as base
 base.BASE=SN/'Q4/05_network_redesign';base.NETWORK_DATA=base.BASE/'02_source_data'
 import q4_river_lake_shared_semicircular_network as network
 import fig5b_top5_driver_responses as b
 b.NETWORK_SOURCE=base.BASE/'09_river_lake_shared/02_source_data';b.ASSOCIATION_SOURCE=SN/'Q4/02_source_data/Q4_all_frozen_candidate_multiscale_associations.csv'
 source=next((S/'原绘图脚本/Q4').rglob('fig5_final_composite.py'));target=helpers/('fig5_final_'+LANG+'.py');shutil.copy2(source,target)
 m=make_module('plot_Q4_'+LANG,target)
 m.BASE=SN/'Q4';m.SOURCE_DIR=SN/'Q4/02_source_data'
 m.SCRIPT_DIR=(SN/'Q4/01_script' if not ZH else Q/'中文重绘/Q4/01_script');m.FIG_DIR=(SN/'Q4/03_figure' if not ZH else Q/'中文重绘/Q4/03_figure');m.QA_DIR=(SN/'Q4/04_QA' if not ZH else Q/'中文重绘/Q4/04_QA')
 if ZH:
  labels=pd.read_csv(D/'Q4_variable_contribution_and_bootstrap_stability.csv').drop_duplicates('column_name').set_index('column_name').variable_name_cn.to_dict()
  def zh_name(col):
   aliases={'fish_mean_total_length':'鱼类平均全长','fish_median_total_length':'鱼类全长中位数','fish_biomass_weighted_total_length':'鱼类生物量加权全长','phytoplankton_chlorophyta_richness':'浮游植物绿藻门丰富度','zooplankton_protozoa_density_pct':'浮游动物原生动物密度比例','zooplankton_protozoa_richness':'浮游动物原生动物丰富度','overall_richness':'综合分类单元丰富度','macrophyte_growth_form_richness':'水生植物生活型丰富度','macrophytes_growth_form_richness':'水生植物生活型丰富度'}
   if col in aliases:return aliases[col]
   text=labels.get(col,col)
   for a,c in [('trait·',''),('beta·',''),('全体·',''),('河流·',''),('湖库·',''),('湖泊·',''),('LCBD局部群落贡献','LCBD'),('平均相异度','平均差异'),('总长度','全长'),('PCoA轴','PCoA'),('Chlorophyta','绿藻门'),('Protozoa','原生动物'),('百分比','比例'),('高型 硅藻','高型硅藻'),('低型 硅藻','低型硅藻'),('·',' ')]:text=text.replace(a,c)
   return text
  b.publication_label=zh_name
 for folder in [m.SCRIPT_DIR,m.QA_DIR,m.FIG_DIR/'PNG',m.FIG_DIR/'PDF',m.FIG_DIR/'SVG',m.FIG_DIR/'TIFF']:folder.mkdir(parents=True,exist_ok=True)
 # Call the original plot functions directly, avoiding a self-copy of an adapter.
 fshare,selected=m.load_panel_data()
 if ZH:selected['display_label']=selected.column_name.map(b.publication_label)
 if ZH:ACTIVE_TRANSLATION=False
 fig,meta=m.plot_fig5(fshare,selected)
 if ZH:
  ACTIVE_TRANSLATION=True
  texts=[x.get_text() for x in fig.findobj(Text) if x.get_text()]
  write_json(Q/'Q4_原图文字用于中文映射.json',sorted(set(texts)))
  for artist in fig.findobj(Text):artist.set_text(artist.get_text())
 removed_notes=[]
 for artist in list(fig.findobj(Text)):
  label=re.sub(r'\s+',' ',artist.get_text()).strip()
  if label.startswith('面积高度：13范围') or label.startswith('Area height = 13-range'):
   removed_notes.append(artist.get_text());artist.remove()
 assert len(removed_notes)==1,removed_notes
 write_json(Q/('图48_删除面积高度说明_'+LANG+'.json'),dict(removed=removed_notes,other_elements_preserved=True,statistical_reanalysis=False))
 if '--qa-only' in sys.argv:
  outputs=[m.FIG_DIR/ext.upper()/('Fig5.'+ext) for ext in ['png','pdf','svg','tiff']]
  qa=m.run_qa(fshare,selected,fig,meta,outputs)
  qa.to_csv(Q/'Fig7_版式与数据QA.csv',index=False,encoding='utf-8-sig')
  assert qa.passed.all(),qa[~qa.passed].to_dict('records')
  plt.close(fig);print('Q4 final QA passed',len(qa),flush=True);sys.exit(0)
 if '--preview' in sys.argv:
  fig.savefig(Q/('Q4_'+LANG+'_预览.png'),dpi=130,bbox_inches='tight',pad_inches=.03,facecolor='white')
  plt.close(fig);print('Q4 preview saved',flush=True);sys.exit(0)
 for ext in ['png','pdf','svg','tiff']:
  opts={'bbox_inches':'tight','pad_inches':.03,'facecolor':'white'}
  if ext in ['png','tiff']:opts['dpi']=600
  if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
  fig.savefig(m.FIG_DIR/ext.upper()/('Fig5.'+ext),**opts)
 plt.close(fig)
 meta['panel_a_summary'].to_csv(SN/'Q4/02_source_data/Fig5_panel_a_ecological_dimension_shares.csv',index=False,encoding='utf-8-sig')
 meta['shared_paths'].to_csv(SN/'Q4/02_source_data/Fig5_panel_c_driver_response_domain_paths.csv',index=False,encoding='utf-8-sig')
 if ZH:finish_zh(Q/'中文重绘/Q4','Q4','图48_具体生物特征对环境梯度的响应及贡献结构')
 print('Q4 '+LANG+' complete',flush=True)
