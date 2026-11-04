from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "10_结果解读报告"
SOURCE_DOCX = REPORT_DIR / (
    "0727_Scale-dependent divergence of water-quality grades and aquatic biotic communities "
    "across rapidly urbanizing river–lake systems_环境子变量13尺度分解扩展版_2026-07-28.docx"
)
OUTPUT_DOCX = REPORT_DIR / (
    "0727_Scale-dependent divergence of water-quality grades and aquatic biotic communities "
    "across rapidly urbanizing river–lake systems_13个递增range修订版_2026-07-29.docx"
)
TAXON_DIR = ROOT / "13_生物类群×五家族精细分析_2026-07-24"
SUBVAR_DIR = ROOT / "14_三类环境子变量×生物五家族尺度分解_2026-07-27"

CN_FONT = "宋体"
EN_FONT = "Times New Roman"


REPLACEMENTS = {
    "1. 水质等级最明确的结果来自地形梯度。": (
        "1. 水质等级最明确的结果来自地形梯度。River 的负相关覆盖全部 13 个 range，其中 "
        "1000–6000 m 获得 FDR 支持；Lake 的负相关在 200–2000 m 获得支持，随后随 range "
        "增大而减弱。城市化总体对应较差水质，但在 13-range 校正后未达到 q≤0.05；"
        "人类直接活动与等级水质没有形成清晰单调轨迹。"
    ),
    "4. 具体生物变量揭示了整体家族信号的组成。": (
        "4. 具体生物变量揭示了整体家族信号的组成。Lake 人类活动与浮游动物密度、总体丰富度、"
        "水生植物与浮游植物多样性在多个相邻 range 呈正相关；Lake 城市化与底栖动物 LCBD、"
        "平均相异度及周转组分在多个相邻 range 呈负相关。River 地形与浮游植物/浮游动物数量和"
        "丰富度负相关，而城市化与鱼类生物量、浮游动物密度及多个功能组成指标相关。"
    ),
    "6. 五家族总体响应可以进一步分解为清晰的类群通道。": (
        "6. 五家族总体响应可以进一步分解为清晰的类群通道。River 的地形响应在 200–3000 m "
        "主要由浮游植物承载，着生藻类在 3500–6000 m 的相对份额上升；城市化响应在 "
        "200–3000 m 以鱼类为主，3500–6000 m 则以着生藻类为主。Lake 的地形和城市化响应在"
        "绝大多数 range 均主要由底栖动物承载，而人类活动响应在全部 13 个 range 均以水生植物"
        "为首要类群，并由浮游植物和浮游动物提供补充证据。"
    ),
    "参数解读边界：参数解读：图中 13 个累积空间范围被分为": (
        "参数解读边界：13 个嵌套累积空间 range 依次为 200、500、1000、1500、2000、2500、"
        "3000、3500、4000、4500、5000、5500 和 6000 m，不预设类别边界。这些 range 是同一"
        "站点的相关空间曲线，不是 13 组独立样本。"
    ),
    "River 地形在 13 个尺度上均为负相关": (
        "River 地形在 13 个 range 上均为负相关，其中 1000–6000 m 的 11 个 range 通过 FDR，"
        "形成连续的相邻-range响应。Lake 地形在 200–2000 m 更强，500 m 达到峰值，2500 m 后"
        "逐渐衰减。因此二者不是“谁更稳健”的区别，而是 River 的响应从 1000 m 延续到 6000 m，"
        "Lake 的统计支持边界止于 2000 m。"
    ),
    "Figure 03.": "Figure 03. 五类水生生物特征对三类环境梯度的13-range响应谱",
    "地形梯度：水质强、生物方向异质。": (
        "地形梯度：水质强、生物方向异质。River 水质 ρ 从 200 m 的 −0.351 增强至 4000 m 的 "
        "−0.646，并在 6000 m 仍为 −0.535；生物家族 RMS 随 range 变化不大，但家族 signed "
        "median 接近 0，说明生物响应并不弱，而是不同变量方向相反。Lake 水质 ρ 在 500 m 达到 "
        "−0.504，200–2000 m 获 FDR 支持，至 6000 m 减弱为 −0.032；生物家族 maxT 支持则很少，"
        "构成端点分离。"
    ),
    "城市化梯度：方向较一致，但生物组成信息更多。": (
        "城市化梯度：方向较一致，但生物组成信息更多。River 水质 ρ 从 200 m 的 0.092 增至 "
        "3000 m 的峰值 0.417，6000 m 为 0.329；生物家族 RMS 约为 0.179–0.195，并有多个家族"
        "获得 maxT 支持。Lake 水质 ρ 从 200 m 的 0.164 增至 4000 m 的峰值 0.366，6000 m 为 "
        "0.150；生物端以 β 多样性在 200–3500 m 最突出。"
    ),
    "统计量与图例说明：颜色表示控制经度": (
        "统计量与图例说明：颜色表示控制经度、纬度、经纬度二次项、经纬度交互和调查次数后的偏 "
        "Spearman ρ。黑色外圈表示同一具体响应在13个range内 BH q≤0.10；白色中心点表示该生物"
        "家族全部响应 × range 的 BH q≤0.05。白底细灰线逐一对应 13 个实际累积 range，不设置"
        "分区色带或类别边界。"
    ),
    "• River 鱼类组成对城市化的峰值为": (
        "• River 鱼类组成对城市化的峰值为 2000 m，R²=0.058，置换 p=0.002，13-range "
        "q=0.0029，且全部 13 个 range 均获得支持。"
    ),
    "• Lake 浮游动物对人类活动的组成中心变化不强": (
        "• Lake 浮游动物对人类活动的组成中心变化不强，但 2500 m 的高—低梯度离散度差为 "
        "−0.096，13-range 离散度 q=0.013。"
    ),
    "River 鱼类能够整合河网连通性": (
        "River 鱼类能够整合河网连通性、河段栖息地和较大范围土地利用，因此地形与城市化均表现为"
        "宽 range 的组成响应。Lake 底栖动物活动范围较小，直接受底质、近岸植被、水深和岸带干扰"
        "影响，其峰值集中在 500–1000 m，提示这些较小累积 range 对局部生境差异更敏感。"
    ),
    "在共同可用响应中，5928 个水体差异检验": (
        "在共同可用响应中，5928 个水体差异检验有 17 个在模块内全部响应 × range 校正后达到 "
        "q≤0.05，57 个达到 q≤0.10。可作生态解释的差异主要包括：3500–6000 m 的地形梯度与"
        "浮游植物绿藻丰富度、浮游动物原生动物丰富度/占比在 River 为负、Lake 为正。"
    ),
    "Figure 09.": "Figure 09. River 生物类群 × 五家族的三环境模块13-range响应图谱",
    "• 地形地貌。 浮游植物的群落基础": (
        "• 地形地貌。浮游植物的群落基础、α 多样性、群落组成和功能/指示群形成最宽的响应带，"
        "其中功能/指示群的 maxT 支持覆盖 200–6000 m，群落组成覆盖 200–5000 m。着生藻类同样"
        "有 4 个家族响应：α 多样性在 3500–6000 m 获得 maxT 支持，β 多样性在 4500–6000 m "
        "获得支持。浮游动物的群落基础从 1500 m 延伸至 6000 m；鱼类的群落基础、组成和功能家族"
        "也有连续响应。"
    ),
    "• 城市化建设。 鱼类 4 个家族": (
        "• 城市化建设。鱼类 4 个家族和着生藻类 3 个家族构成主通道。鱼类群落组成峰值 RMS 为 "
        "0.402（1500 m），maxT 支持 200–4000 m；鱼类 β 多样性峰值 RMS 为 0.374（500 m），"
        "支持 200–3000 m；鱼类功能/指示群覆盖全部 13 个 range。着生藻类的群落组成和功能/指示群"
        "同样覆盖全部 13 个 range，α 多样性主要在 200 m 及 2000–4500 m 获得支持。底栖动物的"
        "组成与功能响应集中于 1500–4500 m，浮游动物的群落基础与功能响应从 1000–1500 m 延伸"
        "至 6000 m。"
    ),
    "生态学解读。 River 的地形响应呈现由悬浮生产者向附着生产者的尺度接力": (
        "生态学解读。River 的地形响应呈现由悬浮生产者向附着生产者的 range 接力：200–3000 m "
        "主要由浮游植物记录，着生藻类在 3500–6000 m 的相对重要性上升。这与河流中局地光照、"
        "河岸遮蔽、坡面输入和更大汇水背景共同塑造初级生产者的过程相符。城市化首先通过鱼类的"
        "组成、β 结构和功能属性表现，并在 range 增至 3500 m 以后由着生藻类与浮游动物继续记录。"
    ),
    "Figure 10.": "Figure 10. Lake 生物类群 × 五家族的三环境模块13-range响应图谱",
    "River 人类活动没有形成广泛 maxT 支持": (
        "River 人类活动没有形成广泛 maxT 支持，因此其份额只描述弱总体效应中的相对构成："
        "200 m 为底栖动物，500–1000 m 为水生植物，1500–3000 m 为鱼类，3500 m 为底栖动物，"
        "4000–4500 m 为水生植物，5000–6000 m 为着生藻类；不宜据此宣称主导生态过程。"
    ),
    "生态学解读。 贡献份额把“尺度效应”从抽象 range": (
        "生态学解读。贡献份额把“尺度效应”落实为 13 个实际 range 上的生物过程接力。River "
        "地形和城市化在相邻 range 间由不同类群接续承载；Lake 的城市化信号在全部 13 个 range "
        "均以底栖动物为主，而人类活动信号随 range 增大总体上更集中到水生植物。"
    ),
    "• 水生植物生活型丰富度与人类活动呈正相关": (
        "• 水生植物生活型丰富度与人类活动呈正相关，中位 ρ=0.421，峰值 ρ=0.502（4000 m）；"
        "Shannon 指数中位 ρ=0.413，峰值 ρ=0.509（4000 m）。两者与水生植物家族在 "
        "2500–6000 m 的连续正响应一致。"
    ),
    "• 浮游动物总密度在 500 m 的峰值": (
        "• 浮游动物总密度在 500 m 的峰值 ρ=0.611，联合支持为 500–1000 m；轮虫密度百分比"
        "在 500 m 的峰值 ρ=0.473，但只具有局部效应量证据。二者共同指向 500–1000 m 的人类"
        "活动响应。"
    ),
    "• 鱼类外来种丰度百分比对人类活动的峰值": (
        "• 鱼类外来种丰度百分比对人类活动的峰值为 ρ=-0.525（500 m），但跨 13 range 中位效应"
        "较小（-0.126），属于局部/分段响应；鱼类周转 LCBD 与城市化在部分相邻 range 呈负相关，"
        "但未获得同等完整的联合支持。Lake 鱼类结论应保持探索性。"
    ),
    "五家族总体响应在类群层面具有明确分工。": (
        "五家族总体响应在类群层面具有明确分工。River 中，地形信号主要由浮游植物、着生藻类、"
        "浮游动物和鱼类共同承载；城市化信号在 200–3000 m 以鱼类为首要贡献类群，3500–6000 m "
        "由着生藻类接续。Lake 中，底栖动物是地形和城市化的主要响应类群，人类活动以水生植物为"
        "核心，并伴随浮游植物 β 重组和浮游动物在 500–1000 m 的数量/功能变化。"
    ),
    "三类环境因素通过不同生物类群进入水生态系统": (
        "三类环境因素通过不同生物类群进入水生态系统，并在每个类群内部作用于不同的群落家族；"
        "这些类群—家族通道在 13 个实际 range 上表现为连续相邻响应、局部相邻响应或方向转折。"
        "River 更突出跨类群的 range 接力，Lake 更突出底栖动物与水生植物分别对城市化和人类活动"
        "的通道分化。"
    ),
    "8A. 三类环境因素的子变量贡献、方向与13尺度分解": (
        "8A. 三类环境因素的子变量贡献、方向与13-range分解"
    ),
    "River 的人口密度在五个家族中均占人类活动模块主导份额": (
        "River 的人口密度在五个家族中均占人类活动模块主导份额：α 多样性 0.770、β 多样性 "
        "0.679、群落基础 0.727、群落组成 0.783、功能/指示 0.736。群落组成和功能/指示在全部 "
        "13 个 range 均获 maxT 支持，说明它们并非由孤立最佳 range 驱动，而是在 200–6000 m "
        "的每个观测 range 上保持统计支持。加权 API 仅在功能/指示家族的 3500–4500 m 获得支持，"
        "属于局部相邻-range通道。"
    ),
    "Figure S14.": "Figure S14. River 五家族的环境子变量13-range响应图谱",
    "Figure S15.": "Figure S15. Lake 五家族的环境子变量13-range响应图谱",
    "统计学说明：编码与 Figure S14 相同。": (
        "统计学说明：编码与 Figure S14 相同。Lake 以 44 个独立 Site 为统计单元。黑色外圈表示"
        "具体“子变量 × range”通过共同置换 maxT p≤0.05；连续外圈表示多个相邻累积 range 保持"
        "统计支持，离散外圈表示局部相邻-range支持。两者均不构成稳健性或地理外推等级。"
    ),
    "Lake 地形响应更具尺度域限制。": (
        "Lake 地形响应集中在部分相邻 range。底栖动物 β 多样性对平均高程的贡献为 0.318，ρ 为 "
        "−0.361，在 200–2000 m 获支持；浮游植物 α 多样性对坡度强度的贡献为 0.365，ρ 为 "
        "+0.302，集中于 4500–6000 m；浮游动物群落基础对相对高程位置的支持集中于 5000–6000 m。"
        "城市化方面，水生植物功能/指示对建筑覆盖在 500–1000 m 获支持；底栖动物 β 多样性对"
        "绿色/自然缓冲呈负向轨迹（1500–2000 m），对道路覆盖呈正向轨迹（1000–1500 m）；"
        "水生植物 α 与群落基础对交通设施的响应出现在约 3000 m。"
    ),
    "统计学说明：点面积为13尺度平均 Johnson": (
        "统计学说明：点面积为13个range的平均 Johnson 贡献份额，颜色为平均 signed median "
        "partial ρ；黑色外圈表示至少一个 range 通过共同置换 maxT p≤0.05。贡献份额仅在同一"
        "环境模块内部可比；不能跨模块相加，也不能将点大小解释为类群生态重要性的绝对排名。"
    ),
    "本扩展分析不将“显著 range 的数量”解释为发现": (
        "本扩展分析不将“显著 range 的数量”解释为发现、稳健或可外推的等级，而是直接报告 "
        "13 个实际 range 上的响应形态。连续相邻-range轨迹表现为方向一致且多个相邻 range 获得 "
        "maxT 支持；分段轨迹只在一个具体的连续range区间获得支持；局部轨迹只在少数相邻 range "
        "出现。转折或正负并存轨迹提示家族内部响应异质、相邻range之间环境含义改变，不能用单一"
        "“最佳 range”概括。"
    ),
    "子变量分解把“水生生物群落的具体特征如何响应三类因素”推进为可检验的生态通道。": (
        "子变量分解把“水生生物群落的具体特征如何响应三类因素”推进为可检验的生态通道。River "
        "中，地形响应由平均高程和地形起伏共同承载，并通过浮游植物、着生藻类及其基础、组成和"
        "功能特征形成跨range接力；城市化响应由道路、建设用地、建筑和绿色/自然缓冲共同构成；"
        "人类活动则主要表现为人口密度对多个类群的连续影响。Lake 中，人口密度主要对应底栖动物 "
        "β 多样性和组成，加权 API 主要对应水生植物的基础、α 多样性、组成和功能，地形及具体"
        "城市结构更多呈现局部相邻-range响应。"
    ),
    "水质等级主要响应地形梯度。River 是 1–6 km": (
        "水质等级主要响应地形梯度。River 在 1000–6000 m 的 11 个相邻 range 获 FDR 支持，"
        "Lake 在 200–2000 m 的 5 个相邻 range 获支持。城市化方向上对应较差水质，但校正后"
        "不确定；人类活动无清晰单调轨迹。"
    ),
    "Lake 城市化主要由底栖动物 β 多样性与组成变化承载": (
        "Lake 城市化主要由底栖动物 β 多样性与组成变化承载，其中道路、绿色/自然缓冲和建筑覆盖"
        "分别在 500–2000 m 的不同相邻 range 形成通道；人类活动表现为人口密度对应底栖动物 β "
        "与组成、加权 API 对应水生植物基础、α 多样性、组成和功能的分化。River 地形由平均高程"
        "和地形起伏共同承载，城市化由道路、建设用地、建筑和绿色/自然缓冲共同构成，人类活动则"
        "以人口密度对多个类群的连续响应为主。"
    ),
    "A further uncertainty concerns the spatial range over which each environmental component is expressed.": (
        "A further uncertainty concerns the spatial range over which each environmental component is expressed. "
        "Conditions adjacent to a waterbody can regulate shading, bank stability and near-channel inputs, whereas "
        "environmental attributes accumulated over progressively larger ranges can alter hydrological connectivity, "
        "source accumulation and dispersal. Because nested cumulative ranges are correlated and have no inherent "
        "categorical boundaries, an apparent response range can be constrained by the set of ranges chosen by the "
        "analyst. A quantitative review of 71 multiscale studies encompassing 583 species found that individual studies "
        "evaluated a mean of only four spatial scales spanning 0.9 orders of magnitude. Estimated scales of effect "
        "occurred at the boundary of the tested range for 44% of species, including 21% at the smallest scale and 23% "
        "at the largest, while only 29% of studies provided a biological rationale for scale selection.[[CITE:16]] Empirical "
        "freshwater studies likewise show that water chemistry, habitat and macroinvertebrates can be related most "
        "strongly to urban cover at different spatial extents, while lake macrophyte responses vary among growth forms "
        "and buffer ranges.[[CITE:17,18]] Freshwater ecosystems are hierarchically connected, so a response observed within one "
        "range can be generated or modified by processes operating beyond it.[[CITE:19]] Urban stream ecology has consequently "
        "identified the coupling of stressors, community responses and management indicators as a persistent research "
        "need.[[CITE:20]] What remains poorly resolved is how water-quality status and multiple components of biological "
        "organisation trace the same environmental gradients across a dense sequence of nested ranges, and whether the "
        "two endpoints identify the same response ranges. Without resolving this scale dependence, apparent agreement "
        "or disagreement between water-quality and biological indicators may partly reflect mismatched analytical ranges "
        "rather than genuine ecological concordance. The selected range may also influence which pressure appears most "
        "important and which range is prioritised for monitoring or intervention."
    ),
    "Here, we examine these questions in Shenzhen, a rapidly urbanised city in southern China": (
        "Here, we examine these questions in Shenzhen, a rapidly urbanised city in southern China where several decades "
        "of land conversion have altered river networks and associated ecosystem services.[[CITE:21]] Over five years, our "
        "research team conducted repeated field surveys and sampled aquatic communities at 86 river sites and 44 lake "
        "sites, yielding 409 river and 155 lake survey records. We analysed these field-collected biological data "
        "alongside ordinal water-quality grades for 19 river and 35 lake spatial units. The biological dataset comprised "
        "313 raw response variables, organised into five response families: community-level attributes, alpha diversity, "
        "beta diversity, community composition, and functional or indicator metrics. These variables represented six "
        "biological groups in rivers and five in lakes. For each waterbody, we quantified natural-landscape conditions, "
        "urban construction and direct human activity at 13 nested cumulative ranges from 200 to 6,000 m. We structured "
        "the study around four linked questions. First, how do water-quality grades in rivers and lakes vary across the "
        "13 ranges in relation to natural-landscape setting (including topographic and geomorphic conditions), urban "
        "construction and direct human activity? Second, how do aquatic communities as a whole respond to these "
        "environmental gradients across the same ranges? Third, to what extent do water-quality and biological response "
        "patterns converge or diverge across environmental gradients and ranges? Fourth, given the integrative role of "
        "aquatic communities in assessing ecosystem health, how do specific characteristics of different biological "
        "groups respond to natural-landscape setting, urban construction and direct human activity?"
    ),
}


def set_run_fonts(run) -> None:
    run.font.name = EN_FONT
    if run.font.size is None:
        run.font.size = Pt(10.5)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), EN_FONT)
    rfonts.set(qn("w:hAnsi"), EN_FONT)
    rfonts.set(qn("w:cs"), EN_FONT)
    rfonts.set(qn("w:eastAsia"), CN_FONT)


def replace_paragraph(paragraph, text: str) -> None:
    for run in list(paragraph.runs):
        paragraph._p.remove(run._r)
    parts = re.split(r"(\[\[CITE:[0-9,]+\]\])", text)
    for part in parts:
        if not part:
            continue
        match = re.fullmatch(r"\[\[CITE:([0-9,]+)\]\]", part)
        run = paragraph.add_run(match.group(1) if match else part)
        set_run_fonts(run)
        if match:
            run.font.superscript = True


def all_paragraphs(doc):
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    for section in doc.sections:
        paragraphs.extend(section.header.paragraphs)
        paragraphs.extend(section.footer.paragraphs)
    return paragraphs


def update_text(doc) -> int:
    changed = 0
    for paragraph in all_paragraphs(doc):
        text = paragraph.text.strip()
        for prefix, replacement in REPLACEMENTS.items():
            if text.startswith(prefix):
                replace_paragraph(paragraph, replacement)
                changed += 1
                break
    for paragraph in all_paragraphs(doc):
        if any(term in paragraph.text for term in ["13尺度", "13 个尺度", "跨尺度"]):
            text = (
                paragraph.text
                .replace("13尺度", "13-range")
                .replace("13 个尺度", "13 个 range")
                .replace("跨尺度", "跨多个range")
            )
            replace_paragraph(paragraph, text)
            changed += 1
    return changed


def set_cell_text(cell, text: str) -> None:
    paragraph = cell.paragraphs[0]
    replace_paragraph(paragraph, text)
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)


def update_tables(doc) -> None:
    set_cell_text(doc.tables[2].cell(0, 6), "13-range FDR q")
    set_cell_text(doc.tables[3].cell(3, 6), "2000 m 峰值；较大 range 仍维持同向效应")
    set_cell_text(doc.tables[3].cell(6, 6), "500–5000 m 多个相邻range正响应")

    table = doc.tables[5]
    headers = ["水体 × 模块", "13 个实际 range 上的首要类群序列", "逐-range统计解读", "统计边界"]
    for j, value in enumerate(headers):
        set_cell_text(table.cell(0, j), value)
    rows = [
        [
            "River × 地形",
            "200–3000 m 多为浮游植物；3500 m 后多为着生藻类（4000 m 除外）",
            "首要贡献类群在3500 m附近交替",
            "不设类别边界",
        ],
        [
            "River × 城市化",
            "200–3000 m 为鱼类；3500–6000 m 为着生藻类",
            "相邻range间由鱼类转为着生藻类",
            "基于逐-range相对贡献",
        ],
        [
            "Lake × 地形",
            "除5000 m为浮游动物外，其余12个range均为底栖动物",
            "底栖动物在绝大多数range保持首位",
            "5000 m贡献接近，不作绝对排序",
        ],
        [
            "Lake × 城市化",
            "13个range均为底栖动物",
            "首位类群不变，贡献份额逐range波动",
            "不等同于因果贡献",
        ],
        [
            "Lake × 人类活动",
            "13个range均为水生植物",
            "份额由200 m的32.8%至6000 m的53.9%，中间并非单调",
            "不等同于生态改善",
        ],
    ]
    for i, values in enumerate(rows, start=1):
        for j, value in enumerate(values):
            set_cell_text(table.cell(i, j), value)

    table = doc.tables[6]
    set_cell_text(table.cell(0, 3), "主要range—生态通道")
    set_cell_text(
        table.cell(1, 3),
        "基础/组成/功能在多个range获支持；浮游植物与着生藻类最清晰",
    )
    set_cell_text(
        table.cell(4, 3),
        "底栖动物β在200–2000 m；部分生产者在4500–6000 m",
    )
    set_cell_text(
        table.cell(5, 3),
        "贡献分散；道路、建筑和缓冲在若干相邻range形成局部响应",
    )


def figure_sources() -> list[Path]:
    sources: list[Path] = []
    for number in range(1, 8):
        matches = list((ROOT / "08_论文级图件" / "PNG").glob(f"Figure{number:02d}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"Figure {number} not uniquely found: {matches}")
        sources.append(matches[0])
    for number in range(8, 13):
        matches = list((TAXON_DIR / "04_论文级图件" / "PNG").glob(f"Figure{number:02d}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"Figure {number} not uniquely found: {matches}")
        sources.append(matches[0])
    for number in range(13, 20):
        matches = list((SUBVAR_DIR / "04_论文级图件" / "PNG").glob(f"FigureS{number}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"Figure S{number} not uniquely found: {matches}")
        sources.append(matches[0])
    return sources


def word_png_bytes(path: Path, max_width: int = 2800) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True, dpi=(300, 300))
    return buffer.getvalue()


def replace_images(doc) -> None:
    sources = figure_sources()
    if len(doc.inline_shapes) != len(sources):
        raise RuntimeError(f"Expected {len(sources)} inline figures, found {len(doc.inline_shapes)}")
    for number, (shape, source) in enumerate(zip(doc.inline_shapes, sources), start=1):
        embed = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        image_part = doc.part.related_parts[embed]
        image_part._blob = word_png_bytes(source)
        shape._inline.docPr.set("title", f"Figure {number}")
        shape._inline.docPr.set(
            "descr",
            f"13个递增累积range（200–6000 m）的科研统计图；不使用三级尺度分区。",
        )


def enforce_fonts(doc) -> None:
    for paragraph in all_paragraphs(doc):
        for run in paragraph.runs:
            if run.text:
                set_run_fonts(run)


def audit(doc) -> None:
    forbidden = [
        "滨水区尺度",
        "中尺度",
        "景观尺度",
        "尺度域",
        "三种背景色",
        "13尺度",
        "riparian zone",
        "intermediate and landscape extents",
        "spatial domains",
    ]
    hits = []
    for paragraph in all_paragraphs(doc):
        for term in forbidden:
            if term in paragraph.text:
                hits.append((term, paragraph.text))
    if hits:
        raise RuntimeError(f"Residual categorical range language: {hits[:20]}")
    if len(doc.inline_shapes) != 19:
        raise RuntimeError(f"Expected 19 figures, found {len(doc.inline_shapes)}")


def main() -> None:
    doc = Document(SOURCE_DOCX)
    changed = update_text(doc)
    update_tables(doc)
    replace_images(doc)
    enforce_fonts(doc)
    audit(doc)
    doc.core_properties.title = "城市水生态13个递增累积range响应分析与论文级解读报告"
    doc.core_properties.subject = "13个累积range（200–6000 m），不设置类别边界"
    doc.save(OUTPUT_DOCX)
    print(
        {
            "output": str(OUTPUT_DOCX),
            "paragraphs_revised": changed,
            "figures_replaced": len(doc.inline_shapes),
            "tables": len(doc.tables),
        }
    )


if __name__ == "__main__":
    main()
