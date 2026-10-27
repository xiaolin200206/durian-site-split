# 多站点前瞻测试：Pahang → Penang → Malacca → Johor（→ Perak）

Seremban 的流程原样保留（`prospective_seremban.py` 不动）。新站点用 `prospective_site.py`，
多了一道**出发前预锁定**，预测和模型哈希在拍照之前就定下来，这才叫前瞻。

每个站点四步，站名用小写：`pahang`、`penang`、`malacca`、`johor`、`perak`。
Pahang 的脚本在 `pahang/`（预锁定已完成，只剩 2–4 步）。下一站的预锁定用 `prelock_next_site.sh`，
`prepare` / `lock` / `score` 把 `pahang/` 里的脚本复制一份、改 `SITE=`。
`prepare` 会按预锁定的树规则自动只收单数号树，文件名里没有树号的会被拒收。

## 周五出发前（Pahang）

- [x] 服务器上换好 v0（2026-10-26，`durian_11cls_640_v5_20261018_420b4c52.onnx`，SHA-256 `420b4c52…1dce`）。
- [x] **预锁定已经做好**：`prospective/pahang_prelock.json`（2026-10-26 13:57 MYT）。
      模型哈希沿用 Seremban 锁定文件里的 200 个；预测 P1–P4 和 Seremban 一字不差；树的分配规则 = 单数号树。
- [ ] **今天**把它推到 GitHub，并把它的 SHA-256 发邮件给自己：
      `git add prospective/pahang_prelock.json prospective/pahang prospective/prospective_site.py prospective/MULTISITE_README.md`
      `git commit -m "Pahang prelock (before visit)" && git push`
- [ ] PWA 设置里打开**分享位置**（默认是关的，不开就没有 GPS）。
- [ ] 把 freeze_v0 里的模型拷到 GPU 机器 `/root/autodl-tmp/durian/pwa/durian_11cls_640_v5_20261018_420b4c52.onnx`（lock 时会核对）。
- [ ] v0 从此不动。Pahang 的照片全部上传之前，服务器上不要 `use` 别的模型。

## 现场

- [ ] 20 棵树，**按走的顺序编号 t1, t2, t3…**，张数不限。每棵树第一张拍树干/树牌/手，当分隔符；树间停 30 秒。
- [ ] **单数号树（t1, t3, t5…）= CEA 前瞻测试；双数号树（t2, t4…）= 会议那条线。**这是预锁定里写死的，不能事后换。
- [ ] 文件名 `p1t1_…`（PWA 导出时按树编号，或回来用 `scripts/prepare/rename_with_tree_id.py`）。
- [ ] 小目标、侧叶、背面顺手多拍——这一站的分数本来就定义为下限。
- [ ] 留 1–2 棵没症状的树。
- [ ] 会议那条线唯一的要求：**PWA 的原始预测别被覆盖**（导出时预测和终标分开）。
- [ ] 记：品种、树龄、喷药周期、天气。

## 回来

- [ ] **所有 Pahang 照片都上传完（App 里待上传 = 0）之前，不要 `use` 新模型。**识别是在上传那一刻做的，
      不是拍照那一刻；没传完就换模型，剩下的照片会被 v1 预标。每张照片的 `model` 字段会记下是谁标的。
- [ ] `admin.py export` 导出后，按你记的每棵树起止时间把文件名改成 `p1t1_…`（导出的文件名只有日期和 id，没有树）。
- [ ] QC：在预标基础上改，**逐图找漏**；记账（加/删/改类各多少）。
- [ ] 每个 `leaf_spot_blight`（和 `web_blight`）框手动改成 **Phomopsis 或 Leaf_rot**——产品模型把这两类合在一起了，
      不改的话 `2_prepare.sh` 会停下。
- [ ] `scale_insect` 预标：叶正面的改成 Psyllid，背面的保留（会被去掉）。
- [ ] 随机 40 张关掉预测框从零标一遍（或外包），另存。
- [ ] 正面白色蜡状小虫 → Psyllid；只有背面能认的才 Scale_insect（反正会被去掉）。
- [ ] 导出 YOLO 文件夹：`images/`、`labels/`、`data.yaml`（12 类名字照 PWA 的即可，脚本会映射）。
- [ ] `bash prospective/pahang/2_prepare.sh` → 看 `pahang_prepare_report.txt`。
- [ ] `3_lock.sh` 填数字 → 跑 → **SHA-256 再发一封邮件 / commit**。
- [ ] `4_score.sh`（GPU）。
- [ ] 结果在 `prospective/pahang_result.txt`、`pahang_per_model.csv`。
- [ ] 这一站的图全部进 v1 的训练集；v1 训好后，Penang 的 `1_prelock.sh` 里 PWA 权重换成 v1。

## 类别映射（脚本里已写好）

PWA 产品模型是 11 类：algal_spot, leaf_spot_blight, leafhopper_damage, pink_disease, psyllid,
psyllid_damage, root_disease, scale_insect, stem_borer, web_blight, weevil_damage。

| PWA 名 | 论文索引 |
|---|---|
| algal_spot | 0 Algal |
| leaf_spot_blight, web_blight | **人工改成 1 Leaf_rot 或 2 Phomopsis**（脚本拒收没改的） |
| psyllid | 3 Psyllid |
| psyllid_damage | 4 Psyllid_damage |
| leafhopper_damage | 5 leaf_hopper_damage |
| scale_insect（叶背）、root_disease、pink_disease、stem_borer、weevil_damage | 框去掉，图保留 |

遇到别的写法脚本会报错停下，在 `ALIASES` 里补一行再跑。

## 文件

| 文件 | 内容 |
|---|---|
| `prospective_site.py` | `prelock` → `prepare` → `lock` → `score` |
| `<site>_prelock.json` | 200 个 LOFO 权重哈希、v0 哈希、P1–P4、采集协议（出发前） |
| `<site>_640/` | 测试集（prepare 生成） |
| `<site>_lock.json` | 预锁定哈希 + 测试集指纹 + QC 记账（打分前） |
| `<site>_per_model.csv`、`<site>_result.txt` | 结果 |

`.gitignore` 里照 Seremban 加四行，让新站的图进仓库：

```
!prospective/pahang_640/images/
!prospective/pahang_640/images/**
!prospective/pahang_640/labels/
!prospective/pahang_640/labels/**
```

## 测试

`prospective_site.py` 的 prelock / prepare / lock 在假数据上跑通过；篡改标签后 score 会拒绝；
重复 prelock、没有 prelock 就 prepare、站名不合法都会被拦下。score 的评估部分和 Seremban 版逐字相同。
