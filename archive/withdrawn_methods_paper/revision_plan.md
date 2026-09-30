# 重构说明与待办

对应文件：`manuscript_nmi_analysis.md`。两份意见（另一位审读 + 我）合并后，我把改动分成三类。

## 一、已经在稿子里改好的（不需要新实验）

| 改动 | 来源 | 位置 |
|---|---|---|
| 标题改为 *Evaluation-unit sampling can dominate reported machine-learning performance*（"accuracy"→"performance"，"is a property"→"can dominate"，因为 GWHD 是反例） | 他 | 标题 |
| 摘要压到 150 词内；正文压到 ~2,900 词（不含 TODO 和表格），Methods ~1,400 词 | 他 | 全文 |
| Related work 并入无标题的开头；Discussion 去掉子标题；Limitations 压成 Discussion 一段，详细版移到 Supplementary Note 4 | 他 + 我 | Main / Discussion |
| Display item 减到 6 个：Table 1 + Fig 1–5；原正文 9 张表移入 Supplementary（对应关系写在稿末 "Supplementary information"） | 他 | 全文 |
| 方差分解的 "Unit" 改名：durian 叫 withheld farm，GWHD/BreaKHis/HAR 叫 **held-out fold / sample composition**，并在 Methods 明说一个 fold 含 9–10 sessions / ~16 patients / 6 subjects | 他 | Results 第 4 节 + Methods |
| 删掉 unit/model 比值列（79、215、3.8）；"unit >> model" 改为 *across the model families and capacities examined, variation in the held-out sample often exceeded variation attributable to the model* | 他 + 我 | Results / Discussion |
| "near-failure to near-perfect" 改为 *wide, practically consequential variation*，HAR 0.756 不再称 near-failure | 他 | 标题句 / Discussion |
| "Nothing predicts" 改为 *Among recorded covariates and the model's own confidence, none gave a stable indicator*；8 个农场的置信度结果明确标为 indication, not a general result | 他 | Results 第 6 节 |
| checkpoint 偏差数字统一：durian 12.3–17.3 点（均值 14.6），GWHD 7.7–8.9 点（均值 8.3）；删掉原来互相矛盾的 "8 to 17 / 8 to 15" | 我 | Results 第 7 节、Fig 5 图注 |
| "Five results" / "Six claims" 统一为 five claims | 我 | Discussion |
| Novelty 重写：不再说 "neither reports dispersion / asks how many groups are enough"，改为承认 clustered-data statistics 已有，本文贡献是 ML 实践中的**幅度测量**（estimand 问题） | 他 + 我 | 开头第 3–4 段、Discussion 第 2 段 |
| 新增引用 25–28：TRIPOD-Cluster、2026 clustered evaluation 论文（待核实）、Bouthillier 2021、Varoquaux 2018；ref 19 改为 YOLO11 仓库 + RT-DETR CVPR 2024 | 他 + 我 | References |
| Fig 1 图注注明 GWHD item-level 线是两模型均值（解释图上 0.658 vs 表中 0.673） | 我 | 图注 |
| 两个 corpus 错误合并为一节，稳定放在 Results 末尾 | 我 | Results 第 8 节 |

## 二、需要跑新分析的（稿子里用 [[TODO]] 标出位置）

按优先级：

1. ~~**弃权分析的 AP50 与主表对不上**~~ **已诊断并部分修好。** 原因不是匹配器也不是种子，是 `abstention_v2.py` 的 `ap50()` 把六个类混成一条曲线算 class-agnostic AP，而 mAP50 是各类 AP 的平均（`matched.append((conf, ok))` 那一行丢了类别）。种子的影响可以排除：rtdetr-l 上种子 42 与五种子均值最大只差 0.041。
   - 置信度和沉默率与类别无关，所以**相关性可以立刻用主表的正确 mAP50 重算**，我已经算了并改进稿子：置信度 vs 得分 **r = +0.054（ρ = −0.095）**，不是原来的 −0.26；沉默率 vs 得分 r = +0.316（ρ = +0.262），不是 +0.61。
   - 结论因此要改写，而且改写后更强也更稳：置信度对哪个农场会失败**完全没有信息**，而且最差的农场（farm 6，0.130）在 0.25 阈值下**从不沉默**，沉默最多的 farm 3（14.5%）反而在中位数以上。原来"符号相反"那句依赖一个 n=8 相关系数的符号，太脆。
   - 还需要重跑的：风险—覆盖曲线（"弃答 50% 使 AP50 从 0.193 升到 0.242"）仍是合并曲线算的，要用 `abstention_v3.py` 重算。该脚本已附：按类算 AP 再对出现过的类平均、默认跑五个种子、`--check` 会逐农场与 `in_region.csv` 对照并在差值 >0.02 时警告。
2. **置信度 vs 单元得分扩展到 GWHD 47 / BreaKHis 81 / HAR 30**（现在优先级最高的一条）。只需要已有 checkpoint 的推理输出（mean max conf / entropy / margin），不用重训。做完后 Fig 4 改成四面板，这一节就从 n=8 的轶事变成跨域结论。
3. **Per-unit 不确定性**：最小 unit 阈值 5/10/20 扫描 + unit 内 item bootstrap。不用重训，只需重跑 per-unit validation。做完后 Fig 1 加第二种误差线。
4. **BreaKHis 分解的类别控制**：现在的 patient fold 按 item 数装箱、没有按类平衡；要么重做 class-balanced fold（需重训 50 runs），要么在单类上重复分解（不需重训）。
5. **方差分解改层次模型**（可选但更好）：unit nested in fold，报 unit-level component。现有 per-unit 表 + fold 归属就能拟合，不需重训。
6. **Nested checkpoint 协议**（另一位审读最看重，但代价最大）：outer unit-disjoint test，inner validation 只从训练单元抽，重训所有 headline 实验（durian 180 + GWHD 59 + BreaKHis 100 runs）。如果做不起，稿子里现在的写法已经是"保留测量、声明为上界"的版本，Methods 相应句子已写好，只要删 TODO。

## 三、投稿前的事务性清理

- 仓库：删 `verify_claims.py*.bak`、`patch_*.py`、`fix_*.py`、`replace_make_figures.py`、空目录 `0.030`；更新 README/HANDOVER 里过时的 run 数和 check 数；`docs/cover_letter.md` 是给 Nature Sustainability 的旧版（五个农场、0.478→0.246），必须重写。
- 作者单位统一（稿子 "Faculty of Computer Science (Data Science)" vs cover letter "Institute of Computer Science and Digital Innovation"）。
- 伦理：拿到学校 determination 后加 ethics statement；填 NMI Reporting Summary。
- 核实 ref 25、26、19 的完整字段。
- 正式稿要嵌入 Fig 1–5 的图，不能只有图注；Fig 1 BreaKHis 面板 "pooled 0.906" 标签被数据点遮住，重绘时挪一下。

## 四、我没有替你做的决定

- 是否做 nested 协议重训（第二节第 6 条）。这是唯一一个"做与不做"会改变稿子定位的选择。
- 是否把 GWHD 的反例放进摘要（我放了一句 "on the fourth, training-seed variance dominates"，因为审稿人迟早会看到，先说比被抓好）。
- 标题两个候选里我用了更强的那个；如果层次模型做完后 unit 份额明显缩水，换成保守版 *Machine-learning performance estimates depend strongly on the evaluation units sampled*。

---

# 补充：Article 路线的决策与新结果（第二轮意见之后）

## 已经做出来的新结果：评价单元 dose-response（零 GPU）

脚本 `eval_unit_curve.py`，输入是已有的 per-unit 表，输出见下。从单元里重复
抽 k 个、按实例加权重算聚合分，4000 次重抽：

| 数据集（单元数，pooled） | k=1 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| Durian bursts (58, 0.293) | 0.326 ±0.247 | 0.308 ±0.119 | 0.297 ±0.082 | 0.297 ±0.053 | 0.295 ±0.029 |
| GWHD sessions (47, 0.539) | 0.486 ±0.125 | 0.520 ±0.080 | 0.533 ±0.060 | 0.538 ±0.040 | 0.538 ±0.020 |
| BreaKHis patients (81, 0.906) | 0.901 ±0.175 | 0.907 ±0.082 | 0.905 ±0.055 | 0.907 ±0.036 | 0.906 ±0.022 |
| HAR subjects (30, 0.951) | 0.950 ±0.051 | 0.951 ±0.024 | 0.951 ±0.015 | 0.951 ±0.009 | — |

90% 区间宽度所需的评价单元数：

| | ≤0.10 | ≤0.05 |
|---|---|---|
| Durian bursts | 32 | 48 |
| GWHD sessions | 23 | 39 |
| BreaKHis patients | 22 | 48 |
| HAR subjects | 3 | 8~9 |

**这构成两条不同的规律，就是第二位审读要的那张主图的一半：**

- 训练单元 ↑：均值可能升（durian +61.5%）也可能不动（GWHD），离散度降 3–6 倍
- 评价单元 ↑：均值不动（四个数据集都稳在第三位小数），离散度按 1/√k 降

Discussion 可以写成：训练单元数决定模型有多好，评价单元数决定你有多知道模型
有多好，两者不能互相替代。原来"应该报告的两个数字"因此多了第三个：报一个
分数时应当配多少个单元。

**必须写进 Methods 的限制**：这些单元的分数来自不同的折，重抽混合了训练集，
是近似而非精确复制。精确版需要固定一个训练集再重抽评价单元 → 要重训。

## Article vs Analysis 的决策

第二位审读把线画在"做不做三件事"，我同意这条线，但三件事代价差很多：

| 要做的事 | 代价 | 建议 |
|---|---|---|
| 评价单元曲线 | 零 GPU，**已完成** | 不论投哪儿都做 |
| 层次方差分解（unit nested in fold，分开 M/T/E/S） | 零 GPU，per-unit 表 + fold 归属即可拟合 | 不论投哪儿都做 |
| 训练单元 × 评价单元二维实验 | GWHD 那一半已有；durian 已有；BreaKHis 是新开销 | 若冲 Article 则做 |
| **Nested checkpoint 协议重训所有 headline** | durian 180 + GWHD 59 + BreaKHis 100 runs | **唯一真正的决策点** |

- 做得起 nested 重训 → 投 Article，标题用 *Evaluation-unit sampling can dominate
  reported machine-learning performance*，Discussion 删掉"reporting convention
  rather than a new method"那句。
- 做不起 → 投 Analysis，checkpoint 那节保留为"测量而非修正"，Methods 明说所有
  单元级数字是上界。现在 `manuscript_nmi_analysis.md` 就是这个版本。

前两项零 GPU，无论如何都该做：做完 Analysis 版也更硬，且 78.7% / 56.1% 那个
命名问题会自动消失。

## 我不同意他的一点

不建议为了 Article 补第五个数据集（音频）。HAR 已经挡住了"这只是视觉重复图像"
的质疑，再加一个的边际收益远低于把统计设计做干净。第二位审读自己也把加数据集
排在优先级最后，这点我们一致。

## Article 版 6 个 display item（若走这条路）

| Display | 内容 |
|---|---|
| Table 1 | 四数据集、单元定义、items、units、models、runs |
| Fig 1 | item vs unit：抬高的均值 + 虚假的精度（现 Fig 1 的合并版） |
| Fig 2 | per-unit 分布 + 跨模型难度排序 + Sabah 面板（地理距离 ≠ 统计距离） |
| Fig 3 | 层次方差分解：evaluation sample / training sample / model / seed |
| Fig 4 | **训练单元 × 评价单元二维实验**（新主图，评价轴已有数据） |
| Fig 5 | checkpoint 复用实验 + provenance case study 面板 |

自曝的三个错误压成一个紧凑小节 *Data provenance errors can silently recreate
unit leakage* + 一个面板，pHash 过程、后缀碰撞、镜头分层全部进 Methods/SI。

---

# 补充二：checkpoint 偏差补齐（2026-09，BreaKHis 完成）

## 权重清点结果

| | last.pt | 结论 |
|---|---|---|
| durian rtdetr-l / 11m / 11n | 各 45 个，齐 | 已有结果 |
| **durian yolo11s** | **best 和 last 都已不存在** | 无法补，需整行重训才有意义；决定不补 |
| GWHD 两模型 | 结果表已齐（59 行） | 已有结果 |
| **BreaKHis 两模型** | **100 个全齐** | **已跑完，零训练** |
| HAR | 不适用 | sklearn 固定 300 迭代 / 森林无早停，本来就干净 |
| dose-response（durian k01–k06、GWHD k02–k38） | last.pt 都在 | 将来若要做无选择版本，材料现成 |

## BreaKHis 新结果（`checkpoint_bias_bkh.py`，200 次验证，约 40 分钟）

| model | regime | best | last | delta |
|---|---|---|---|---|
| yolo11n-cls | item-level | 0.9915 | 0.9905 | +0.0010 |
| yolo11n-cls | unit-level | 0.9126 | 0.8879 | +0.0247 |
| yolo11s-cls | item-level | 0.9931 | 0.9918 | +0.0012 |
| yolo11s-cls | unit-level | 0.9063 | 0.8771 | +0.0292 |

高估幅度：yolo11n-cls 8.0% → **10.4%**（+2.4 点），yolo11s-cls 8.7% → **11.6%**（+2.8 点）。
无选择协议下错误率 0.82% → 12.29%，**15.0 倍**（原为 13.5 倍）。

## 三个数据集合起来的新说法

| 数据集 | 选择后 | 无选择 | 升高 |
|---|---|---|---|
| durian（3 模型） | 37.4–42.7% | 51.6–58.2% | 12.3–17.3 点 |
| GWHD（2 模型） | 19.6–22.4% | 28.5–30.1% | 7.7–8.9 点 |
| BreaKHis（2 模型） | 8.0–8.7% | 10.4–11.6% | 2.4–2.8 点 |

**新观察，值得写进 Results**：选择偏差的大小跟着划分效应本身走，三个数据集同序。
机制上说得通——被留出的单元越难，每轮 fitness 在它上面波动越大，"挑最好一轮"
能捡的便宜越多；BreaKHis 的病人折接近天花板，可挑的余地最小。

七个 dataset-model 组合方向全部一致，item-level 都几乎不动（0.001–0.011），
unit-level 都下降。稿子里 checkpoint 那一节和 Fig 5 图注已按此改写，nested
协议的 TODO 已删除——七个组合的测量已经界定了上界有多松，不需要重训。

## 还要做的两件小事

1. 把 `results_v2/checkpoint_bias.csv` 复制进仓库 `results_breakhis/`，
   并在 `verify_claims.py` 的 `breakhis()` 里加检查（best/last 四个均值、
   两个高估幅度、item-level 变动 < 0.002）。
2. Limitations 补一句：durian YOLO11s 的两种权重都未保留，该组合缺席；
   所测七个组合方向一致。HAR 不需要此项修正，理由写进 Methods。
