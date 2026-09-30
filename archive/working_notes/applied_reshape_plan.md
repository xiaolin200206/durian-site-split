# 重塑计划：从"聚类基准的方差"到"一个新果园要付多少代价"

2026-09-28 · 取代 PR-D-26-13571（已撤）

## 一句话

旧版问的是"held-out 分数属于模型还是属于单元"，为了普适性拉了三个别的学科。
新版只问榴莲：**检测器拿到一个没见过的果园能有多准，要收多少个果园的数据才够，
到了新果园拍几张照片能补回多少。** 主语是果园和设备，不是 benchmark。

## 暂定题目

- *What a new orchard costs: farm-level generalisation of durian disease and pest detection, and how much field data closes the gap*
- 备选：*More farms or more photos? Data budgets for on-device durian disease and pest detection*

## 目标期刊

**Computers and Electronics in Agriculture**（Elsevier，hybrid，走订阅通道免 APC）。
备选 Precision Agriculture（Springer，hybrid）。

## 三个研究问题

| | 问题 | 数据状态 |
|---|---|---|
| RQ1 | 到一个没见过的果园，准确率掉多少？掉在哪些病虫害上？ | **已有**（clean 协议 LOFO + 新写的逐类分析） |
| RQ2 | 训练数据该花在更多果园还是每个果园更多照片上？ | **要跑**：`farm_budget.py` |
| RQ3 | 到新果园先拍 5/10/20 张做校准，能补回多少？微调和重训哪个划算？ | **要跑**：`new_farm_calibration.py` |

## 已经有的数字（全部来自 clean 协议，可以直接用）

**RQ1 — 新果园的代价**
- yolo11n：随机切分 0.428 → 留一果园 0.223 mAP50（−47.8%）。五个检测器 −47.8% 到 −55.8%，Faster R-CNN −39.3%。
- 8 个果园逐个评：0.094–0.356（各模型平均），六个检测器对果园难度的排序几乎一致（Spearman ρ = 0.87），都把同两个果园排最后。
- 模型对结果几乎没贡献：果园分量 82.4%，模型 0.0%。durian 上任意两个检测器的平均配对差 ≤ 0.022 mAP50。→ **选模型看延迟和功耗就行**（接 TII 那篇：yolo11n 上 Pi 5）。
- Sabah（1,800 km 外、另一个岛）0.253–0.281，不比隔壁果园（0.214–0.236）差。→ 难度单位是"新果园"，不是距离。
- 模型置信度不能预警难果园：置信度和果园分数 ρ = +0.09；8 个采集协变量也都不预测。→ **设备自己不知道它在一个难果园上**，这是 RQ3 存在的理由。

**RQ1 新增 — 逐类（`results_applied/per_class_transfer.csv`，五个检测器平均）**

| 类别 | 出现的果园 | 随机切分 AP50 | 新果园 AP50 | 保留 |
|---|---|---|---|---|
| leaf_hopper_damage | 8 | 0.592 | 0.349 | 59% |
| Phomopsis | 3 | 0.460 | 0.266 | 58% |
| Algal | 7 | 0.504 | 0.272 | 54% |
| Leaf_rot | 8 | 0.488 | 0.236 | 48% |
| Psyllid | 7 | 0.410 | 0.125 | 30% |
| Psyllid_damage | 6 | 0.337 | 0.087 | 25% |

→ 病害掉一半，**木虱（Psyllid）和木虱危害只剩四分之一到三分之一**。虫子小、形态随果园变，
新果园代价主要落在虫害上。这一条旧版没有，是应用版的新发现。

**方法上的坑（写成部署建议，不写成方法论贡献）**
- 用 held-out 果园做 Ultralytics 的 val split 选 checkpoint：yolo11n 的高估从 47.8% 被报成 37.4%，少报 10.4 个点。→ 建议：val 从训练果园里抽，或固定 epoch。
- 按文件名 join 果园 ID 错了 8.9%（49 张 farm 6 的图标成 farm 0），靠 perceptual hash 找回。→ 建议：采集 app 拍照时直接写入果园 ID。

## 要跑的两个实验

### RQ2：farm_budget.py

- 留一果园（8 个），训练果园数 k ∈ {1, 2, 4, 7}，每果园张数 m ∈ {15, 50, all}
- k < 7 抽 2 组果园组合，种子 42 和 1，yolo11n
- 同一组果园的三个 m 用嵌套图片（15 ⊂ 50 ⊂ all），配对比较
- **协议**：held-out 果园不参与任何训练决策；不留内层验证、不早停、固定约 2,000 步、评 last.pt
- 核心对比：**同样 ~100 张，7 个果园 × 15 张 vs 2 个果园 × 50 张**（还有 ~200、~400 两档）
- 336 run，每种子约 12–15 GPU 小时（4090 估算）。只跑种子 42 就能出图

### RQ3：new_farm_calibration.py（等 RQ2 的 k=7 m=all 跑完）

- 基座 = RQ2 里 k=7、m=all 的模型（其余 7 个果园全部图）
- 新果园按拍摄顺序对半切：前半校准池、后半测试，中间丢 3 张缓冲；再反过来一次（两个方向）
- 同一棵树的连拍不会一半校准一半测试（顺序切，只有一个边界）
- 三个 arm：base（不校准）/ ft（冻结 backbone 微调，~300 步）/ rt（其余 7 果园 + 校准图从头重训）
- m ∈ {5, 10, 20, all}
- ft 128 run（~1–2 分钟一个）+ rt 64 run（~5 分钟一个），合计约 8–10 GPU 小时

### 命令（AutoDL）

```bash
cd /root/autodl-tmp/durian
cp <repo>/results_durian/split_assignment.csv .      # 没有的话
cp <repo>/scripts/applied/*.py .

# RQ2
python farm_budget.py --step build
python farm_budget.py --step train --limit 2 --epochs-cap 3   # 先试两个，确认能跑
screen -S fb
python farm_budget.py --step train --yes --seeds 42           # 先跑一个种子
python farm_budget.py --step eval  --seeds 42
#   → results_applied/farm_budget.csv 发回来
python farm_budget.py --step train --yes --seeds 1            # 第二个种子
python farm_budget.py --step eval

# RQ3
python new_farm_calibration.py --step build [--meta metadata_backup.csv]
python new_farm_calibration.py --step train --yes
python new_farm_calibration.py --step eval
#   → results_applied/new_farm_calibration.csv 发回来
```

`--limit` 和 `--epochs-cap` 只用于试跑；加了 `--epochs-cap` 的 run 自动写到
`fb_runs_trial/`，不会和正式结果混。

有 EXIF 时间表（旧的 `metadata_backup.csv`）就给 RQ3 加 `--meta`，
没有就按相机计数器排序（已处理 9999→0000 回绕）。

## 新文章结构（CEA 格式，约 6,000–7,000 词）

1. **Introduction** — 设备卖给新果园；论文报的数字是在见过的果园上测的；没有公开的榴莲/多数植病数据集带果园 ID；三个问题。
2. **Data and device** — 8 个半岛果园 + 2 个 Sabah 果园，827 + 281 张，六类，采集和标注；果园 ID 的 perceptual-hash 复核（压成一段）；设备和 yolo11n 的选择（引 TII）。
3. **Evaluation protocol** — 留一果园、held-out 果园不进任何训练决策、为什么（一段 + checkpoint 泄漏的 10.4 点作为理由）。
4. **Results**
   - 4.1 新果园的代价（总体 + 逐果园 + 逐类）— 图 1、表 2
   - 4.2 模型不是瓶颈（五个检测器差 ≤ 0.02；果园 82% / 模型 0%）— 一段 + 小图
   - 4.3 更多果园还是更多照片（RQ2）— 图 2
   - 4.4 新果园校准（RQ3）— 图 3
   - 4.5 跨岛与预警（Sabah；置信度不能预警）— 一段
5. **Discussion — deployment recommendations** — 采集预算怎么花；上手校准怎么设计；报数字要报留一果园；拍照时写果园 ID。
6. **Limitations** — 8 个果园、一个采集者、标注一致性未测、离线评估。
7. **Conclusions**

## 从旧稿删掉的

GWHD、BreaKHis、UCI HAR 三个数据集；estimand 定义；两套方差分解的公式和 bootstrap 层级讨论；
ICC / 设计效应；57 个 roster 子集；18 对模型 bootstrap（留一句）；四项 reporting checklist
（改写成部署建议）；unitcheck（降为 Code availability 里一句）。

## 和已发/在审稿件的关系

同一批图。和 IVC 那篇（capture session、分类）不同：检测、果园单位、新增 RQ2/RQ3。
和旧 PR 稿不同：只剩 durian、问题不同、两个新实验。投稿时在 cover letter 里写明撤稿及原因。
