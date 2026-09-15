#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_protocol_breakhis.py -- 外层留出患者完全不参与任何训练决策，
外加一个真正跨架构族的分类器。

两件事在一个脚本里，因为它们必须在同一个协议下跑，否则不可比。

[1] 干净协议
    原 splits/<cfg>/  是 train/<class>/ 与 val/<class>/ 的软链树，val 就是
    被留出的那批患者，而训练循环每个 epoch 都在它上面算 fitness。这里把
    外层患者从训练树里彻底拿掉：

        clean_splits/<cfg>/train/<class>/   外层训练患者的 90% 图像
        clean_splits/<cfg>/val/<class>/     同一批患者的另外 10%（内层）
        外层患者                            训练全程不出现，最后评一次

    ★ 内层验证按患者切，不按图像切。 同一患者的图像高度相关，若按图像
      随机切 10%，内层验证里的图几乎都能在内层训练里找到同一张片子的邻
      居，早停信号会虚高且没有意义。所以从外层训练患者里整人整人地抽
      约 10% 的图像量作内层验证，并保持良恶性比例。

[2] 跨架构族模型
    现有两个模型是 YOLO11n-cls 与 YOLO11s-cls，同族不同容量。审读多次
    指出，模型方差份额低可能只是因为比较的是近乎相同的模型。用 timm 的
    ImageNet 预训练模型加线性头，训练循环与 Ultralytics 完全不同，才算
    真正换了族。默认 efficientnet_b0。

    这个模型走同一套 clean_splits，同样只看内层验证集早停，同样在最后
    评一次外层。

用法：
    pip install timm
    cd /root/autodl-tmp/breakhis
    python clean_protocol_breakhis.py --step splits
    python clean_protocol_breakhis.py --step train                  # YOLO 两个
    python clean_protocol_breakhis.py --step train --yes
    python clean_protocol_breakhis.py --step train_timm --yes       # 跨族
    python clean_protocol_breakhis.py --step eval
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

ROOT = "/root/autodl-tmp/breakhis"
INNER_FRAC = 0.10
INNER_SEED = 20260911
YOLO_MODELS = {"yolo11n-cls": ("yolo11n-cls.pt", 256),
               "yolo11s-cls": ("yolo11s-cls.pt", 256)}
SEEDS = [42, 1, 2, 3, 4]


def P(root):
    return {"root": root,
            "proc": os.path.join(root, "processed"),
            "splits": os.path.join(root, "splits"),
            "clean_splits": os.path.join(root, "clean_splits"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_clean")}


def manifest(p):
    f = os.path.join(p["proc"], "manifest_clean.csv")
    if not os.path.isfile(f):
        sys.exit(f"找不到 {f}")
    return list(csv.DictReader(open(f, encoding="utf-8-sig")))


def link_tree(dst, items):
    """items: [(split, label, image_id, image_path)]"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    n = Counter()
    for s, lab, iid, ip in items:
        d = os.path.join(dst, s, lab)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, iid + ".png")
        if not os.path.exists(link):
            try:
                os.symlink(ip, link)
            except OSError:
                shutil.copy2(ip, link)
        n[s] += 1
    return n


# ----------------------------------------------------------------- splits --
def step_splits(a, p):
    rows = manifest(p)
    print(f"manifest {len(rows)} 行, "
          f"{len({r['patient'] for r in rows})} 个患者")
    print(f"\n内层验证按患者整人抽，目标约 {INNER_FRAC:.0%} 的图像量，"
          f"保持良恶性比例（固定种子 {INNER_SEED}）\n")
    print(f"  {'fold':<18}{'outer imgs':>11}{'inner train':>13}"
          f"{'inner val':>11}{'inner val pts':>14}")

    meta = {}
    for regime, col in (("random", "random_fold"),
                        ("patient", "patient_fold")):
        for f in sorted({r[col] for r in rows}, key=int):
            name = f"{regime}_fold{f}"
            outer = [r for r in rows if r[col] == f]
            train_rows = [r for r in rows if r[col] != f]

            # 外层训练行按患者分组；整人抽进内层验证，按标签分层
            by_pat = defaultdict(list)
            for r in train_rows:
                by_pat[r["patient"]].append(r)
            lab_of = {q: v[0]["label"] for q, v in by_pat.items()}
            target = INNER_FRAC * len(train_rows)

            rng = random.Random(INNER_SEED)
            inner_pats = set()
            for lab in sorted({v for v in lab_of.values()}):
                pool = sorted([q for q in by_pat if lab_of[q] == lab])
                rng.shuffle(pool)
                quota = INNER_FRAC * sum(len(by_pat[q]) for q in pool)
                got = 0
                for q in pool:
                    if got >= quota and inner_pats:
                        break
                    inner_pats.add(q)
                    got += len(by_pat[q])

            inner_val = [r for r in train_rows if r["patient"] in inner_pats]
            inner_tr = [r for r in train_rows
                        if r["patient"] not in inner_pats]

            # 外层患者不得出现在内层任何一侧（random 折按图划分，
            # 患者本就跨折，所以这里只对 patient 折断言）
            if regime == "patient":
                op = {r["patient"] for r in outer}
                assert not (op & {r["patient"] for r in inner_tr}), name
                assert not (op & {r["patient"] for r in inner_val}), name
            oid = {r["image_id"] for r in outer}
            assert not (oid & {r["image_id"] for r in inner_tr}), name
            assert not (oid & {r["image_id"] for r in inner_val}), name

            items = ([("train", r["label"], r["image_id"], r["image_path"])
                      for r in inner_tr]
                     + [("val", r["label"], r["image_id"], r["image_path"])
                        for r in inner_val])
            n = link_tree(os.path.join(p["clean_splits"], name), items)
            meta[name] = {"outer": len(outer), "inner_train": n["train"],
                          "inner_val": n["val"],
                          "inner_val_patients": len(inner_pats)}
            print(f"  {name:<18}{len(outer):>11}{n['train']:>13}"
                  f"{n['val']:>11}{len(inner_pats):>14}")

    os.makedirs(p["clean_splits"], exist_ok=True)
    json.dump({"inner_frac": INNER_FRAC, "inner_seed": INNER_SEED,
               "inner_split_level": "patient", "folds": meta},
              open(os.path.join(p["clean_splits"], "split_meta.json"), "w"),
              indent=1)
    print(f"\n写到 {p['clean_splits']}")
    print("外层 val 仍在原 splits/ 下，训练时完全不引用。")


# ------------------------------------------------------------- YOLO train --
def step_train(a, p):
    from ultralytics import YOLO
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isdir(os.path.join(p["clean_splits"], d,
                                                "train")))
    jobs = [(m, c, s) for m in a.models for s in a.seeds for c in cfgs]
    todo = [j for j in jobs if not os.path.isfile(os.path.join(
        p["runs"], f"clean_{j[0]}_{j[1]}_s{j[2]}", "weights", "best.pt"))]
    print(f"配置 {len(cfgs)}，模型 {a.models}，种子 {a.seeds}")
    print(f"共 {len(jobs)} run，待跑 {len(todo)}")
    print(f"epochs {a.epochs}  patience {a.patience}  imgsz {a.imgsz}  "
          f"batch {a.batch}")
    print("★ 训练只看内层验证集；外层患者一次都不出现。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return
    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"clean_{model}_{cfg}_s{seed}"
        if os.path.isfile(os.path.join(p["runs"], name, "weights",
                                       "best.pt")):
            continue
        el = (time.time() - t0) / 60
        eta = (el / done * (len(todo) - done) / 60) if done else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        w, b = YOLO_MODELS[model]
        try:
            YOLO(w).train(data=os.path.join(p["clean_splits"], cfg),
                          imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                          batch=a.batch or b, workers=a.workers,
                          patience=a.patience, project=p["runs"], name=name,
                          exist_ok=True, deterministic=True, plots=False,
                          verbose=False)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
    print(f"\n结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# ------------------------------------------------------- cross-family (timm)
def _loaders(root, imgsz, batch, workers, seed):
    import torch
    from torchvision import datasets, transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    va_tf = transforms.Compose([
        transforms.Resize(int(imgsz * 1.14)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    g = torch.Generator()
    g.manual_seed(seed)
    tr = datasets.ImageFolder(os.path.join(root, "train"), tr_tf)
    va = datasets.ImageFolder(os.path.join(root, "val"), va_tf)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,
                                        num_workers=workers, generator=g,
                                        drop_last=False),
            torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False,
                                        num_workers=workers),
            tr.classes)


def _train_one_timm(arch, split_dir, seed, a, out_dir):
    import torch
    import timm
    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, classes = _loaders(split_dir, a.imgsz, a.batch or 64,
                               a.workers, seed)
    model = timm.create_model(arch, pretrained=True,
                              num_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_ep, bad = -1.0, -1, 0
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        ok = n = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                ok += (model(x).argmax(1) == y).sum().item()
                n += y.numel()
        acc = ok / max(1, n)
        torch.save({"model": model.state_dict(), "classes": classes,
                    "epoch": ep, "inner_acc": acc},
                   os.path.join(out_dir, "last.pt"))
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            shutil.copy2(os.path.join(out_dir, "last.pt"),
                         os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                break
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"    epoch {ep:>3}  inner acc {acc:.4f}"
                  f"  best {best:.4f}@{best_ep}", flush=True)
    with open(os.path.join(out_dir, "epochs.txt"), "w") as fh:
        fh.write(f"{ep + 1}\n")
    return best, best_ep, ep + 1


def step_train_timm(a, p):
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isdir(os.path.join(p["clean_splits"], d,
                                                "train")))
    jobs = [(c, s) for s in a.seeds for c in cfgs]
    print(f"跨族模型 {a.arch}（timm, ImageNet 预训练）")
    print(f"配置 {len(cfgs)}，种子 {a.seeds}，共 {len(jobs)} run")
    print(f"epochs {a.epochs}  patience {a.patience}  imgsz {a.imgsz}  "
          f"batch {a.batch or 64}  lr {a.lr}")
    print("★ 与两个 YOLO 分类器走同一套 clean_splits，同样只看内层验证集。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return
    t0 = time.time()
    for i, (cfg, seed) in enumerate(jobs, 1):
        name = f"clean_{a.arch}_{cfg}_s{seed}"
        out = os.path.join(p["runs"], name, "weights")
        if os.path.isfile(os.path.join(out, "best.pt")):
            continue
        print(f"\n[{i}/{len(jobs)}] {name}   已用 "
              f"{(time.time()-t0)/60:.0f} 分钟", flush=True)
        b, be, ne = _train_one_timm(
            a.arch, os.path.join(p["clean_splits"], cfg), seed, a, out)
        print(f"  完成：内层 best {b:.4f} @ epoch {be}，共 {ne} 轮")
    print(f"\n结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# ------------------------------------------------------------------- eval --
def _eval_timm(arch, ckpt, outer_dir, a):
    import torch
    import timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    classes = st["classes"]
    model = timm.create_model(arch, pretrained=False,
                              num_classes=len(classes)).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    _, va, _ = _loaders(outer_dir, a.imgsz, a.batch or 64, a.workers, 0)
    from torchvision import datasets
    ds = datasets.ImageFolder(os.path.join(outer_dir, "val"))
    paths = [q for q, _ in ds.samples]
    probs, preds, ys = [], [], []
    with torch.no_grad():
        for x, y in va:
            o = torch.softmax(model(x.to(dev)), 1).cpu()
            probs += o.tolist()
            preds += o.argmax(1).tolist()
            ys += y.tolist()
    acc = sum(int(a_ == b_) for a_, b_ in zip(preds, ys)) / max(1, len(ys))
    return acc, classes, paths, probs, preds, ys


def step_eval(a, p):
    """外层评估：训练结束后第一次也是唯一一次看外层患者。"""
    rows = manifest(p)
    os.makedirs(p["results"], exist_ok=True)
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isdir(os.path.join(p["clean_splits"], d,
                                                "train")))
    # 外层树：只有 val
    outer_root = os.path.join(p["results"], "_outer")
    for regime, col in (("random", "random_fold"),
                        ("patient", "patient_fold")):
        for f in sorted({r[col] for r in rows}, key=int):
            name = f"{regime}_fold{f}"
            d = os.path.join(outer_root, name)
            if os.path.isdir(os.path.join(d, "val")):
                continue
            items = [("val", r["label"], r["image_id"], r["image_path"])
                     for r in rows if r[col] == f]
            # ImageFolder 需要 train 也存在，链同一批，只读 val
            link_tree(d, items + [("train", r["label"], r["image_id"],
                                   r["image_path"])
                                  for r in rows if r[col] == f])

    out_rows, pred_rows = [], []
    from ultralytics import YOLO
    for model in a.models:
        for cfg in cfgs:
            for seed in a.seeds:
                name = f"clean_{model}_{cfg}_s{seed}"
                for tag, wf in (("best", "best.pt"), ("last", "last.pt")):
                    w = os.path.join(p["runs"], name, "weights", wf)
                    if not os.path.isfile(w):
                        continue
                    r = YOLO(w).val(data=os.path.join(outer_root, cfg),
                                    imgsz=a.imgsz, split="val",
                                    batch=a.batch or 256, workers=a.workers,
                                    project=p["runs"],
                                    name=f"{name}_outer_{tag}",
                                    exist_ok=True, verbose=False, plots=False)
                    out_rows.append({"model": model, "config": cfg,
                                     "seed": seed, "checkpoint": tag,
                                     "top1": float(r.top1)})
                    print(f"  {name} [{tag}]  top1 {float(r.top1):.4f}",
                          flush=True)

    if a.arch:
        for cfg in cfgs:
            for seed in a.seeds:
                name = f"clean_{a.arch}_{cfg}_s{seed}"
                for tag in ("best", "last"):
                    ck = os.path.join(p["runs"], name, "weights", f"{tag}.pt")
                    if not os.path.isfile(ck):
                        continue
                    acc, classes, paths, probs, preds, ys = _eval_timm(
                        a.arch, ck, os.path.join(outer_root, cfg), a)
                    out_rows.append({"model": a.arch, "config": cfg,
                                     "seed": seed, "checkpoint": tag,
                                     "top1": acc})
                    print(f"  {name} [{tag}]  top1 {acc:.4f}", flush=True)
                    if tag == "best":
                        for ip, pr, pd, yy in zip(paths, probs, preds, ys):
                            pred_rows.append({
                                "model": a.arch, "config": cfg, "seed": seed,
                                "image": os.path.splitext(
                                    os.path.basename(ip))[0],
                                "true": classes[yy], "pred": classes[pd],
                                "max_prob": max(pr),
                                "probs": ";".join(f"{x:.5f}" for x in pr)})

    if out_rows:
        f = os.path.join(p["results"], "in_region_clean.csv")
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w_.writeheader()
            w_.writerows(out_rows)
        print(f"\n写出 {f}  ({len(out_rows)} 行)")
    if pred_rows:
        f = os.path.join(p["results"], "outer_predictions.csv")
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
            w_.writeheader()
            w_.writerows(pred_rows)
        print(f"写出 {f}  ({len(pred_rows)} 行)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True,
                    choices=["splits", "train", "train_timm", "eval"])
    ap.add_argument("--models", nargs="+",
                    default=["yolo11n-cls", "yolo11s-cls"])
    ap.add_argument("--arch", default="efficientnet_b0",
                    help="timm 跨族模型；eval 时传空字符串可跳过")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(os.path.abspath(a.root))
    {"splits": step_splits, "train": step_train,
     "train_timm": step_train_timm, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
