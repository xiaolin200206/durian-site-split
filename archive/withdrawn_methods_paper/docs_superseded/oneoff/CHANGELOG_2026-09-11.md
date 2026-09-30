# 这一轮改了什么

## 新结果

**BreaKHis checkpoint 选择偏差**（`results_breakhis/checkpoint_bias.csv`）
100 个 run 的 last 权重重新验证，零训练。item-level 几乎不动（+0.0010 /
+0.0012），unit-level 明显下降（+0.0247 / +0.0292）。高估幅度 8.0% → 10.4%、
8.7% → 11.6%。无选择协议下错误率 0.82% → 12.29%，15.0 倍。

至此三个数据集、七个 dataset-model 组合全部测过，方向一致：durian 升高
12.3–17.3 点，GWHD 7.7–8.9，BreaKHis 2.4–2.8。选择偏差的大小跟着划分效应
本身走，这是机制预测的结果。**因此不做 nested 协议重训**——七个组合的实测
已经界定了上界有多松。

**评价单元 dose-response**（`results_eval_unit_curve.csv`，`eval_unit_curve.py`）
零 GPU，从 per-unit 表重抽。等权口径下均值在每个 k 上都不动，离散度按 1/√k
下降。与训练单元曲线构成两条不同的规律：训练单元买模型质量，评价单元只买
测量精度。90% 区间宽度 ≤0.10 所需单元数：durian 32、GWHD 13、BreaKHis 24、
HAR 3。

**置信度结论修正**（尚未重跑，但相关系数已可更新）
`abstention_v2.py` 的 `ap50()` 把六个类混成一条曲线算 class-agnostic AP，
而 mAP50 是各类 AP 的平均，同一个农场差到 0.31。置信度与类别无关，所以用
主表的正确 mAP50 重算：置信度 vs 得分 **r = +0.054**（原报 −0.26），沉默率
vs 得分 r = +0.316（原报 +0.61）。结论从"符号相反"改为"完全没有信息"，更弱
但更稳；最差的 farm 6 从不沉默这个例子仍然成立。风险—覆盖曲线要等
`abstention_v3.py` 重跑。

## CI

263 → **322 条**，全绿。新增：评价单元曲线 41 条、BreaKHis checkpoint 18 条。
其中两条锁的是论点而非数字：`k=16 mean unchanged from k=1`（评价单元只买
精度）和 `spread falls as 1/sqrt(k)`（带有限总体修正）。

## 新增文件

```
eval_unit_curve.py                     评价单元重抽，零 GPU
abstention_v3.py                       修正版弃权分析，尚未跑
scripts/analyse/checkpoint_bias_bkh.py BreaKHis checkpoint 偏差，已跑完
results_breakhis/checkpoint_bias.csv   100 行
results_eval_unit_curve.csv            派生表
docs/manuscript_nmi_analysis.md        重构后的稿子（NMI Analysis 规格）
docs/revision_plan.md                  改了什么、还剩什么
docs/cover_letter.md                   重写，投 NMI
```

## 清理

- 删 `verify_claims.py.bak` / `.gwhd.bak` / `.split.bak`、
  `scripts/analyse/make_figures.py.bak`、空目录 `0.030`
- 一次性脚本移入 `scripts/oneoff/`：`fix_model_split.py`、`fix_thresholds.py`、
  `patch_gwhd_k2.py`、`patch_unitcheck.py`、`replace_make_figures.py`、
  `update_gwhd_checks.py`、`merge_results.py`、`merge_tables.py`
- 旧 cover letter 改名为 `cover_letter_nature_sustainability_OLD.md`
  （它用的是五个农场、0.478→0.246 的旧数字，与现稿全部矛盾）

## 还没做

1. `abstention_v3.py --check` 重跑，更新风险—覆盖那两个数（0.193 → 0.242）
2. 置信度分析扩到 GWHD 47 / BreaKHis 81 / HAR 30 —— 剩下最有价值的一件，
   用已有 checkpoint 推理即可。现在 durian 的 r = +0.05 只是 n=8 的观察
3. 层次方差分解（unit nested in fold），纯计算。做完 BreaKHis 的 78.7% 和
   HAR 的 56.1% 才能正当地叫 unit component
4. `HANDOVER.md` 和 `README.md` 里的 run 数、check 数已过时
5. durian YOLO11s 的 checkpoint 偏差：best 和 last 权重都已不存在，需整行
   重训才有意义，决定不补。Limitations 需注明
