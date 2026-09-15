# 以 NCS 审稿人视角的逐字审读记录

干净协议重跑之后，稿子是重写的而不是改的。这份记录的是重写之后五轮自查
找到的问题。全部已改，并在可能的地方写成了 CI 断言。

## 第一轮：数字与范围

| # | 问题 | 实情 | 处理 |
|---|---|---|---|
| 1 | 正文写「eight model configurations across four architecture families」 | 实际是十个配置、五个族（YOLO、DETR、R-CNN、MLP、forest） | 改为十与五，CI 锁住配置数 |
| 2 | 每单元范围用的是单个模型的极值，却引用跨模型的 Supp Table 8 | durian 写 0.079–0.406（yolo11l），表里是 0.094–0.356（跨模型均值）；GWHD、HAR 同样 | 全部改成跨模型均值，四个数据集的上下界各写一条 CI |
| 3 | 「Between-unit heterogeneity is 98% and 94%」 | 噪声占方差 1.7% 与 5.6%，换算成 s.d. 是 99% 与 97% | 改正，CI 锁住 |
| 4 | HAR 的错误率由 macro F1 反推 | 1 − macro F1 不是错误率；用 accuracy 得到 1/635 与 1/4.8 | 改用 accuracy，CI 明确注明来源 |
| 5 | Table 1 把 HAR 的 100 个 run 列为「clean runs」 | HAR 没有 checkpoint 选择，原始 run 本就满足干净协议，没有重跑 | 加星号与脚注 |
| 6 | 「Every dataset was trained on every fold with every seed」 | Faster R-CNN 三种子、GWHD 三种子，且 Faster R-CNN 不进方差分解 | 改写为「each model on its own schedule」并说明分解用哪五个 |
| 7 | 三协议对照声称「eight of eight」 | durian yolo11s 的 last 权重未保留，只有七个组合 | 改为七，CI 锁住 |

## 第二轮：全称词与机制断言

**「Every figure below is from that protocol」是假的。** 训练单元 dose-response、
协变量与置信度分析、burst 阈值扫描三项都来自被污染的协议，Methods 里也这么
写了，正文与 Methods 自相矛盾。改为逐项点名哪三项不是，并在每项出现处加注。

**这是同一类错误的第三次出现**（前两次是「All figures come from one code path」
和「the split rule is the only difference」）。共同点是一句概括跑在证据前面，
而 CI 抓不到，因为它没有对应的可重算数值。

「they determine whether a benchmark table can be read at all」改为
「bear directly on how a benchmark table should be read」。

「The gap measures how much a model can exploit the near-duplication」把一个
解释写成了已证实的机制；一个两阶段检测器对五个一阶段的、单一语料的比较支撑
不起「measures」。改为明确标注为 hypothesis。

## 第三轮：正文与补充材料的数值一致性

抽出正文所有三位小数，逐一在补充材料里查。十五个不匹配项全部核实为引用编号、
DOI 或表内行（重抽曲线的宽度值在 Supp Table 9 的表格里，只是没有逐个复述）。
无实质矛盾。

## 第四轮：图、图注、表

图注里的每个数值都能在 `summary_*.csv` 里找到；五个图注对应五个图片文件，
编号无缺无余。这一轮没有发现问题，因为新的绘图脚本只从 summary 表取数，
不再有只存在于绘图代码里的数字。

## 第五轮：结构与 Discussion

Discussion 无子标题（NCS 要求）、参考文献 29 条（上限约 50）、Results 九个
主题小标题。

**Discussion 里仍写着「94–98% of that spread」**，而 Results 已改成 97–99%。
同一个量在两处不一致，正是 `cross_audit.py` 该抓的类型——所以给它的概念表
补了五条（异质性份额、单元分量、可分辨配对数、高估幅度范围、d 的范围）。

## 结果

- `verify_claims.py`：**142 / 142**
- `cross_audit.py`：**0 项需人工确认**
- 正文 3,380 词、摘要 146、display item 6、参考文献 29，全部在 NCS 限内
- 冒号揭示句 17 → 7，`rather than` 12 → 6，词汇层面无 AI 标记

## 仍未解决

伦理判定与 NCS reporting summary，两项都在外部。

---

# 第二份 NCS 视角审读（2026-09-15）—— 三条主要质疑的处理

## 第二条：方差分解的不确定性 —— 已用数据回答

`scripts/analyse/decomposition_robustness.py`，零 GPU，三项检查全部写进
Supp Table 14 并锁进 CI：

**(a) Bootstrap 区间。** 单元分量 46–93%，模型分量上界 0.1–8.3%。区间宽，
但每个数据集上单元分量的下界都高于模型分量的上界。摘要与正文改为报区间。

**(b) 固定效应。** durian 上固定效应与随机效应各分量差在 1 点以内。结论不
依赖「模型是总体的样本」这个假设——这正是审读质疑的地方。

**(c) 模型集合子集重抽。** durian 六个配置的全部 57 个子集，单元份额 70–93%，
模型份额从未超过 1.1%。族数越多单元份额越低（1 族 90%，3 族 77%），方向
如审读所料，但三个族仍留四分之三给单元。**趋势承认了质疑，量级挡住了质疑。**

明确写出它不能外推到未训练的架构。

## 第三条：翻转率不是样本量要求 —— 已改措辞

正文明写重抽在有限池内，翻转率是「现有比较的诊断」而非「未来 benchmark 的
样本量建议」。Methods 补写正态近似的三个假设（配对差近似正态、样本方差估计
总体方差、1.645 为单侧 5% 临界值）、无有限总体修正，以及它与模拟的偏差
（大池 ±4 点、八农场池 ±9 点）。报告的是模拟值，闭式解只作为 d 的标度指引。

## 第一条：普遍性 —— 收窄措辞，未加数据集

标题改为 *Held-out unit variability overshadows architectural differences in
clustered machine-learning benchmarks*（审读建议之一）。摘要「against 0–3% for
the model」加「configurations tested」的限定。加数据集需要重新开机，未做。

## 其余采纳的点

- Faster R-CNN 的较小高估不是比率分母假象：绝对降幅 14.4 分对一阶段的
  20.5–27.8 分，正文并列报告，CI 锁住
- 「污染协议是稻草人」的防御：正文明写这是训练框架的默认行为，任何把留出组
  交给 trainer 当验证集的研究都会继承它；同时明说本文不量化它在文献中的
  流行程度。**没有点名具体论文**，因为无法从摘要核实哪篇确实这么做了，指名
  而错是更大的风险
- 现象命名为 checkpoint-selection leakage，与 item-level leakage 区分
- 语料错误段改为 provenance audit 的口吻，不再自贬
- estimand 段加「benchmark leaderboards report the first」
- 摘要结尾加行动导向：inner validation set drawn from training units

## 结果

`verify_claims.py` 162/162，`cross_audit.py` 0 项，正文 3,49x 词、摘要 150。

---

# 第三份 NCS 视角审读（2026-09-15）—— 三条都成立，一条是实质错误

## 第二条：噪声分解的指标不匹配 —— **确认是错误**

稿子在 HAR 上报 macro F1，却把两个分类数据集的单元得分都当二项均值做噪声
分解。**macro F1 不是正确分类比例，方差不是 p(1−p)/n。** 这个错误是本轮引入
的，审读抓得准。审读还指出第二层：即便换成准确率，单元内观测也不独立
（同一病人的近重复切片、同一受试者的相邻窗口），二项方差会低估噪声。

**改法**（`scripts/analyse/noise_and_inference_fix.py`）：

- HAR 改用 accuracy 并在 Methods 明写为什么不能用 macro F1
- 两个数据集各给**两个**估计。第一个用五个种子的重复评估——同一单元被五次
  独立训练的权重各评一次，其间的方差直接测「重复测量的不稳定性」，不对
  单元内部结构作任何假设。第二个是二项估计，捕捉单元内抽样但假设独立。
- **两者朝相反方向偏**：病人上重复估计更大（3.2% 对 1.8%），受试者上二项
  估计更大（5.5–5.9% 对 0.1–0.4%）。取较大者作保守读法。

结论数值上稳住了：噪声上限 3.2%（病人）与 5.9%（受试者），异质性 98% 与
97%。但现在有了不依赖独立性假设的支撑路径，并明写真正的块 bootstrap 需要
逐 item 预测，现有表不带。

## 第一条：池内稳定性与总体推断混用 —— **确认是内部矛盾**

无放回抽样抽满全池时反转率必为零，所以「某对需要 63 个 session 而只有 47
个」在池内定义下不成立。原稿把总体判据的数说成了池内的事。

拆开报之后差别巨大：**池内判据下 18 对全部可达**（durian 的 yolo11n vs
yolo11s 只要 7 个农场），**总体判据下只有 4 对**（同一对需要 26 个）。同一对
的两个数最大差 10 倍以上。

正文改为分两段问两个问题，Methods 写清两者定义与假设，Supp Table 16 并列
两列，并明说「an earlier version of this manuscript conflated them」。摘要
改为「fourteen have paired differences too small for the available units to
resolve」，去掉「only four separate reliably」这个混用的说法。

## 第三条：dose-response 仍受污染 —— **降级，未重跑**

审读正确指出「所有组用同一协议」不能保证偏差大小相同，且低 k 端的抽样筛选
偏差（小麦 k=2 时 8 次意图抽样只活 1 次）系统性偏向大单元。

干净协议重跑需要约 30 GPU-小时。本轮不做，改为**降级**：正文不再把它当作
结果，只说「evaluation units 那一半是确定的，training units 那一半预算未
在干净协议下验证」；Fig 3 的 a、b 面板在图注里标为 preliminary 并列出两条
限制；Methods 的小节改名为 "Dose-response (preliminary)"。

如果审稿人坚持，这是一个明确的补充实验，脚本框架已就位。

## 结果

`verify_claims.py` **172/172**（新增 20 条，其中「池内 18 对全可达而总体只有
4 对」和「两判据在某对上差一个量级」两条锁的是定义区分本身）。
`cross_audit.py` 0 项。正文 3,498 词、摘要 149。

---

# 第四份 NCS 视角审读（2026-09-15）—— 三条统计质疑，结论主动收窄

这一轮的处理原则变了：不再补实验，而是把三条主张收到证据能支撑的强度。
理由写在最后。

## 第一条：五个折不足以支撑区间（表1、补充表14）

审读指出五折设计下「对折 bootstrap」只有 5 个可抽对象，且各折训练集共享
60% 的单元。改为**对单元重采样**（30–81 个，设计上可交换的层级）：

| | 对折 bootstrap | 对单元 bootstrap |
|---|---|---|
| GWHD 单元分量 | [46.0, 86.6] | **[44.4, 74.9]** |
| BreaKHis | [79.6, 92.8] | **[75.0, 88.8]** |
| HAR | [65.1, 81.8] | **[45.0, 77.4]** |

单元级区间在两个数据集上更宽。**分离仍然成立**——单元下界 44–75% 高于
模型上界 0.2–8.2%。正文改报单元级区间，并明写训练集重叠未被消除、
区间可能仍偏窄、重复分组划分加重训练才能settle。

## 第二条：噪声估计不是上界（补充表15）—— 比审读说的更严重

取二项与跨种子重复的较大者，仍不覆盖组内抽样误差。用设计效应
deff = 1 + (m̄−1)·ICC 做敏感性：

| | 平均单元大小 | 噪声吃掉一半方差所需 ICC |
|---|---|---|
| BreaKHis | 98 张 | **0.26** |
| HAR | 343 窗口 | **0.02** |

**HAR 上 ICC 只要超过 0.022，噪声就占一半方差**，而同一受试者的相邻活动
窗口不可能有这么低的相关。所以「97% 是真实异质性」在 HAR 上几乎肯定不成立。

结论改为条件式：在病人上稳健（任何合理的片内相关都撑得住），在受试者上
「conditional on a correlation low enough that adjacent windows from one
wearer are unlikely to satisfy it」。直接估 ICC 需要逐 item 预测，已发布的
per-unit 表不带。

## 第三条：所需单元数未传播 d 的不确定性（补充表16）

对单元 bootstrap 得到 d 的分布，再代入 (1.645/d)²：

| 配对 | N | d | d 的 90% CI | 所需单元数 CI |
|---|---|---|---|---|
| GWHD 两检测器 | 47 | 0.21 | [0.03, 0.58] | **[9, 3948]** |
| durian 各对 | 8 | — | — | 中位宽度 **2,221** |

点估计落在跨三个数量级的区间里。**「十四对需要比语料更多的单元」整句撤回。**

**换了一个不需要外推的主判据**：对单元 bootstrap 配对差，看 90% 区间是否
含零。这不假设分布、不外推、不把小 d 平方取倒数放大误差。结果是
**18 对里 4 对的区间不含零**——三个是同族容量对，第四个是 HAR 的感知机对
随机森林（本研究中唯一一对除输入外毫无共享的模型）。图像数据集上没有一对
跨族比较被分辨出来。

顺带修正一处：初稿写「四对中没有一对跨族」，但 MLP vs 随机森林是跨族的，
CI 抓到了，已改。

## 为什么这一轮不补实验

前四轮意见都来自 LLM 模拟审稿，不是真实审稿人。它们只能读文本、没有取舍
成本、彼此高度相关（同一模型的多次采样）。抓到的真错误必须改——macro F1
当二项均值、池内与总体混用、d 的不确定性未传播，这三条不管谁提都得改。但
继续按模拟意见迭代的边际收益在下降，而真实审稿人可能关心完全不同的事
（数据代表性、可获得性、对实践者的可操作性）。

因此这一轮把三条主张收到证据能支撑的强度，投出去，把进一步的加强留给真实
审稿意见指路。已识别但未做的补强，按代价排序记录在案：

1. **逐 item 预测 + 块 bootstrap**（BreaKHis 约 1 GPU-小时，HAR 几分钟 CPU）
   —— 能把噪声结论从条件式变成确定，是性价比最高的一项
2. **BreaKHis 重复分组划分**（约 6 GPU-小时）—— 检验区间覆盖率
3. **dose-response 干净协议重跑**（约 30 GPU-小时）—— 已在上一轮降级
4. **加异构架构到 GWHD/BreaKHis** —— 扩大模型族覆盖

## 结果

`verify_claims.py` **185/185**（新增 13 条，锁住三条收窄后的主张，包括
「配对差区间不含零的只有 4 对」「HAR 在 ICC 0.05 时噪声过半」「d 的转换
不作为估计报告」）。`cross_audit.py` 0 项。正文 3,483 词、摘要 150。
