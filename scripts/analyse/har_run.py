#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
har_run.py -- UCI HAR 上的同一套分析。第四个数据集，第一个非图像模态。

前三个都是图像。这一个是可穿戴传感器的时序窗口，用来检验主张是否只是
"分层采集的图像"的性质，还是"分层采集的数据"的性质：

    榴莲      图像   检测   农业   8 农场
    GWHD      图像   检测   农业   47 会话
    BreaKHis  图像   分类   医学   81 患者
    HAR       时序   分类   行为   30 受试者   ← 非图像

评估单元是受试者。同一个人的连续窗口共享步态、身高、手机佩戴方式和当次
录制的一切，随机划分会把它们分到边界两侧 —— 与 capture burst、acquisition
session、patient 是同一个机制。

UCI HAR 的官方划分本来就是按受试者分的（21 训练 / 9 测试），也就是说这个
领域已经承认要分组。但据我们所知，没有人报告过受试者之间的离散度有多大，
或者需要多少个受试者报告值才稳定。本脚本给出这两个量。

做四件事，与前三个数据集同口径：
  [1] 随机 5 折 vs 受试者互斥 5 折，各 5 seed
  [2] 逐受试者评估：每个受试者由没见过他的那一折评分
  [3] 剂量曲线：训练窗口总数固定，只让受试者数变
  [4] 泄漏断言

分类器用 scikit-learn 的 MLP，561 维特征直接输入。数据小，全套几分钟。

用法：
    python har_run.py --step download --root /root/autodl-tmp/har
    python har_run.py --step build    --root /root/autodl-tmp/har
    python har_run.py --step train    --root /root/autodl-tmp/har
    python har_run.py --step per_subject --root /root/autodl-tmp/har
    python har_run.py --step site_count  --root /root/autodl-tmp/har
"""

import argparse
import csv
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")
MIRROR = ("https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2F"
          "UCI%20HAR%20Dataset.zip")
SEEDS = [42, 1, 2, 3, 4]
FOLDS = 5
ACTS = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
        4: "SITTING", 5: "STANDING", 6: "LAYING"}


def sh(c):
    print("$ " + c, flush=True)
    return subprocess.run(c, shell=True).returncode


def step_download(a):
    os.makedirs(a.root, exist_ok=True)
    z = os.path.join(a.root, "har.zip")
    if not os.path.isfile(z):
        if sh(f'wget -c -O "{z}" "{MIRROR}"') != 0:
            print("镜像失败，试 UCI 原站")
            if sh(f'wget -c -O "{z}" "{URL}"') != 0:
                return 1
    sh(f'cd "{a.root}" && unzip -o -q har.zip')
    # UCI 原站的包里可能还嵌一层 zip
    for dp, _, fns in os.walk(a.root):
        for f in fns:
            if f.endswith(".zip") and f != "har.zip":
                sh(f'cd "{dp}" && unzip -o -q "{f}"')
    print("解压完成")
    return 0


def find_har(root):
    for dp, dns, fns in os.walk(root):
        if "features.txt" in fns and "train" in dns and "test" in dns:
            return dp
    return None


def read_col(p):
    with open(p) as fh:
        return [line.strip() for line in fh if line.strip()]


def step_build(a):
    src = find_har(a.root)
    if src is None:
        sys.exit(f"在 {a.root} 下找不到 UCI HAR Dataset（应含 features.txt）")
    print(f"数据根: {src}")

    rows = []
    for part in ("train", "test"):
        X = os.path.join(src, part, f"X_{part}.txt")
        y = os.path.join(src, part, f"y_{part}.txt")
        s = os.path.join(src, part, f"subject_{part}.txt")
        for p in (X, y, s):
            if not os.path.isfile(p):
                sys.exit(f"缺 {p}")
        ys, ss = read_col(y), read_col(s)
        with open(X) as fh:
            for i, line in enumerate(fh):
                v = line.split()
                if not v:
                    continue
                rows.append({"idx": len(rows), "subject": int(ss[i]),
                             "label": int(ys[i]), "source": part,
                             "x": [float(t) for t in v]})
    n_feat = len(rows[0]["x"])
    subs = sorted({r["subject"] for r in rows})
    print(f"窗口 {len(rows)}   受试者 {len(subs)}   特征 {n_feat}")
    print(f"类别: {dict(Counter(ACTS[r['label']] for r in rows))}")
    print(f"官方划分: train {sum(1 for r in rows if r['source']=='train')} "
          f"/ test {sum(1 for r in rows if r['source']=='test')}"
          f"  （本身就是按受试者分的，21/9）")

    # 每个受试者的窗口数
    c = Counter(r["subject"] for r in rows)
    print(f"每受试者窗口数: 最少 {min(c.values())}  中位 "
          f"{int(st.median(list(c.values())))}  最多 {max(c.values())}")

    rnd = random.Random(a.seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    for pos, i in enumerate(idx):
        rows[i]["random_fold"] = pos % a.folds
    # 受试者折：贪心装箱平衡窗口数
    bins, load = [[] for _ in range(a.folds)], [0] * a.folds
    for sub, n in c.most_common():
        j = load.index(min(load))
        bins[j].append(sub)
        load[j] += n
    sf = {s: j for j, ss_ in enumerate(bins) for s in ss_}
    for r in rows:
        r["subject_fold"] = sf[r["subject"]]

    os.makedirs(os.path.join(a.root, "processed"), exist_ok=True)
    mp = os.path.join(a.root, "processed", "manifest.csv")
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "subject", "label", "activity", "source",
                    "random_fold", "subject_fold"])
        for r in rows:
            w.writerow([r["idx"], r["subject"], r["label"],
                        ACTS[r["label"]], r["source"],
                        r["random_fold"], r["subject_fold"]])
    import numpy as np
    np.save(os.path.join(a.root, "processed", "X.npy"),
            np.array([r["x"] for r in rows], dtype="float32"))
    np.save(os.path.join(a.root, "processed", "y.npy"),
            np.array([r["label"] for r in rows], dtype="int64"))
    print(f"写出 {mp} 与 X.npy / y.npy")

    print(f"\n  {'fold':<6}{'random val':>12}{'subject val':>13}"
          f"{'subjects':>10}")
    for f in range(a.folds):
        rv = sum(1 for r in rows if r["random_fold"] == f)
        sv = [r for r in rows if r["subject_fold"] == f]
        print(f"  {f:<6}{rv:>12}{len(sv):>13}"
              f"{len({r['subject'] for r in sv}):>10}")

    print("\n" + "=" * 66)
    print("泄漏断言")
    print("=" * 66)
    bad = 0
    for f in range(a.folds):
        tr = {r["subject"] for r in rows if r["subject_fold"] != f}
        va = {r["subject"] for r in rows if r["subject_fold"] == f}
        if tr & va:
            bad += 1
    print(f"  受试者跨折: {'无' if bad == 0 else f'★ {bad}'}")
    span = defaultdict(set)
    for r in rows:
        span[r["subject"]].add(r["random_fold"])
    n_span = sum(1 for v in span.values() if len(v) > 1)
    print(f"  随机划分下，窗口分散到多折的受试者: {n_span} / {len(span)} "
          f"({100*n_span/len(span):.0f}%)")
    print("  与前三个数据集同一个结构：随机划分让几乎每个单元同时出现在")
    print("  训练和验证两侧。")
    return 0


# ------------------------------------------------------------------ train --
def load_data(root):
    import numpy as np
    p = os.path.join(root, "processed")
    X = np.load(os.path.join(p, "X.npy"))
    y = np.load(os.path.join(p, "y.npy"))
    with open(os.path.join(p, "manifest.csv"), newline="") as fh:
        man = list(csv.DictReader(fh))
    return X, y, man


def fit_eval(Xtr, ytr, Xte, yte, seed, hidden=256, iters=300):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=iters,
                      random_state=seed, early_stopping=False)
    m.fit(sc.transform(Xtr), ytr)
    p = m.predict(sc.transform(Xte))
    return (accuracy_score(yte, p),
            f1_score(yte, p, average="macro"), m, sc)


def step_train(a):
    import numpy as np
    X, y, man = load_data(a.root)
    seeds = a.seeds or SEEDS
    rows = []
    for regime, col in (("random", "random_fold"),
                        ("subject", "subject_fold")):
        for f in sorted({m[col] for m in man}, key=int):
            te = np.array([i for i, m in enumerate(man) if m[col] == f])
            tr = np.array([i for i, m in enumerate(man) if m[col] != f])
            for s in seeds:
                acc, f1, _, _ = fit_eval(X[tr], y[tr], X[te], y[te], s)
                rows.append({"model": "mlp256", "config": f"{regime}_fold{f}",
                             "seed": s, "n_train": len(tr), "n_val": len(te),
                             "accuracy": acc, "macro_f1": f1})
                print(f"  {regime}_fold{f}_s{s:<3} acc {acc:.4f}  "
                      f"F1 {f1:.4f}", flush=True)
    out = os.path.join(a.root, "results_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "in_region.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 66)
    print("  随机划分 vs 受试者互斥")
    print("=" * 66)
    means = {}
    for reg in ("random", "subject"):
        g = defaultdict(list)
        for r in rows:
            if r["config"].startswith(reg):
                g[r["config"]].append(r["macro_f1"])
        fm = [st.mean(v) for v in g.values()]
        within = st.mean([st.stdev(v) for v in g.values() if len(v) > 1])
        means[reg] = st.mean(fm)
        print(f"  {reg:<9} macro F1 {st.mean(fm):.4f}  fold sd "
              f"{st.stdev(fm):.4f}  seed sd {within:.4f}  n={len(fm)}")
    o = 100 * (means["random"] - means["subject"]) / means["random"]
    print(f"\n  overstatement {o:.1f}%"
          f"   (durian 44.3/41.1%, GWHD 17.6%)")
    print(f"  写出 {out}/in_region.csv")
    return 0


def step_per_subject(a):
    """每个受试者单独评估，用没见过他的那一折。"""
    import numpy as np
    X, y, man = load_data(a.root)
    seeds = a.seeds or SEEDS
    by_sub = defaultdict(list)
    fold_of = {}
    for i, m in enumerate(man):
        by_sub[m["subject"]].append(i)
        fold_of[m["subject"]] = m["subject_fold"]

    rows = []
    for f in sorted({m["subject_fold"] for m in man}, key=int):
        tr = np.array([i for i, m in enumerate(man)
                       if m["subject_fold"] != f])
        for s in seeds:
            _, _, mdl, sc = fit_eval(X[tr], y[tr], X[tr[:10]], y[tr[:10]], s)
            from sklearn.metrics import accuracy_score, f1_score
            for sub, idxs in by_sub.items():
                if fold_of[sub] != f:
                    continue
                ii = np.array(idxs)
                p = mdl.predict(sc.transform(X[ii]))
                rows.append({"group": sub, "n_windows": len(ii),
                             "fold": f"subject_fold{f}", "seed": s,
                             "accuracy": accuracy_score(y[ii], p),
                             "macro_f1": f1_score(y[ii], p,
                                                  average="macro")})
        print(f"  fold {f} done", flush=True)

    out = os.path.join(a.root, "results_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "har_by_subject.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    per = defaultdict(list)
    nw = {}
    for r in rows:
        per[r["group"]].append(r["macro_f1"])
        nw[r["group"]] = r["n_windows"]
    m_ = {g: st.mean(v) for g, v in per.items()}
    v = list(m_.values())
    mu, sd = st.mean(v), st.stdev(v)
    print("\n" + "=" * 66)
    print(f"  逐受试者（{len(v)} 个单元，每个由未见过他的模型评分）")
    print("=" * 66)
    print(f"  均值 {mu:.4f}  sd {sd:.4f}  CV {sd/mu:.3f}  "
          f"范围 {min(v):.3f}-{max(v):.3f}  极差 {max(v)/min(v):.1f}x")
    print(f"\n  {'subject':<10}{'windows':>9}{'macro F1':>10}")
    for g in sorted(m_, key=lambda x: m_[x]):
        print(f"  {g:<10}{nw[g]:>9}{m_[g]:>10.4f}")
    print("\n  对照: durian per-burst CV 0.75, GWHD per-domain CV 0.31")
    return 0


def step_site_count(a):
    """训练窗口总数固定，只让受试者数变。"""
    import numpy as np
    X, y, man = load_data(a.root)
    rnd = random.Random(a.seed)
    by_sub = defaultdict(list)
    for i, m in enumerate(man):
        by_sub[m["subject"]].append(i)
    subs = sorted(by_sub)
    hold = rnd.sample(subs, a.holdout)
    pool = [s for s in subs if s not in hold]
    te = np.array([i for s in hold for i in by_sub[s]])
    print(f"held-out 受试者 {a.holdout} 个 / {len(te)} 窗口: {sorted(hold)}")
    print(f"可抽 {len(pool)} 个受试者")

    ks = [k for k in a.ks if k <= len(pool)]
    n_train = a.n_train
    rows = []
    for k in ks:
        for d in range(a.draws):
            sub = rnd.sample(pool, k)
            avail = [i for s in sub for i in by_sub[s]]
            if len(avail) < n_train:
                print(f"  k={k} d={d}: 只有 {len(avail)}，跳过")
                continue
            per = max(1, n_train // k)
            pick = []
            for s in sub:
                v = by_sub[s][:]
                rnd.shuffle(v)
                pick += v[:per]
            rnd.shuffle(pick)
            pick = pick[:n_train]
            tr = np.array(pick)
            for s_ in (a.seeds or [42, 1]):
                acc, f1, _, _ = fit_eval(X[tr], y[tr], X[te], y[te], s_)
                rows.append({"k": k, "draw": d, "seed": s_,
                             "n_train": len(tr), "accuracy": acc,
                             "macro_f1": f1,
                             "subjects": ";".join(map(str, sorted(sub)))})
            print(f"  k={k} d={d}  F1 "
                  f"{st.mean([r['macro_f1'] for r in rows[-2:]]):.4f}",
                  flush=True)

    out = os.path.join(a.root, "results_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "site_count.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 66)
    print(f"  held-out 受试者上的表现 vs 训练受试者数"
          f"   (训练窗口固定 {n_train})")
    print("=" * 66)
    byk = defaultdict(list)
    bykd = defaultdict(lambda: defaultdict(list))
    for r in rows:
        byk[r["k"]].append(r["macro_f1"])
        bykd[r["k"]][r["draw"]].append(r["macro_f1"])
    print(f"  {'k':>4}{'runs':>6}{'mean':>9}{'draw sd':>10}{'seed sd':>10}")
    for k in sorted(byk):
        dm = [st.mean(v) for v in bykd[k].values()]
        ws = [st.stdev(v) for v in bykd[k].values() if len(v) > 1]
        print(f"  {k:>4}{len(byk[k]):>6}{st.mean(byk[k]):>9.4f}"
              f"{(st.stdev(dm) if len(dm) > 1 else 0):>10.4f}"
              f"{(st.mean(ws) if ws else 0):>10.4f}")
    ks_ = sorted(byk)
    lo, hi = st.mean(byk[ks_[0]]), st.mean(byk[ks_[-1]])
    print(f"\n  k={ks_[0]} -> k={ks_[-1]}: {lo:.4f} -> {hi:.4f} "
          f"({100*(hi-lo)/lo:+.1f}%)")
    print("  训练窗口总数不变，唯一变的是它们来自几个受试者。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["download", "build", "train",
                             "per_subject", "site_count"])
    ap.add_argument("--root", default="/root/autodl-tmp/har")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 24])
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--holdout", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=1000)
    a = ap.parse_args()
    return {"download": step_download, "build": step_build,
            "train": step_train, "per_subject": step_per_subject,
            "site_count": step_site_count}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
