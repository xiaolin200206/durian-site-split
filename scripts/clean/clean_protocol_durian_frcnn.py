#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_protocol_durian_frcnn.py -- durian 上的两阶段检测器，干净协议。

为什么要它：现在 durian 的跨族对照是 RT-DETR 对 YOLO，但两者都是一阶段
端到端检测器，说是"跨族"有点勉强。Faster R-CNN 是两阶段、有 RPN、有锚框，
用 torchvision 的实现和自己的训练循环，和 Ultralytics 没有任何共用代码。
模型比较那一节（哪些配对差稳、哪些脆）因此多一个真正远的对照。

协议与 clean_protocol_durian.py 完全一致：
  - 读同一套 clean_splits/<cfg>/{train,val}.txt
  - 早停与 checkpoint 选择只看内层验证集
  - 外层农场训练全程不出现，最后评一次
  - best 与 last 都存，外层评估存逐图预测

★ 指标口径必须和主表一致。这里用 torchmetrics 的 MeanAveragePrecision，
  它实现的是 COCO 口径的 mAP@0.5，与 Ultralytics 的 mAP50 是同一个定义。
  不一致的话整条比较就废了，所以脚本跑完会打印一行提示要人工核对。

用法：
    pip install torchmetrics
    cd /root/autodl-tmp/durian
    python clean_protocol_durian_frcnn.py --step train            # 看清单
    screen -S frcnn
    python clean_protocol_durian_frcnn.py --step train --yes
    python clean_protocol_durian_frcnn.py --step eval
"""

import argparse
import csv
import os
import random
import sys
import time

ROOT = "/root/autodl-tmp/durian"
ARCH = "fasterrcnn_resnet50_fpn_v2"


def P(root):
    return {"root": root,
            "splits": os.path.join(root, "splits"),
            "clean_splits": os.path.join(root, "clean_splits"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_clean")}


def read_lines(f):
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip()]


def class_names(root):
    import yaml
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns
                  if d not in ("runs", "splits", "clean_splits",
                               "results_v2", "results_clean")]
        if "merged_peninsula" in dns:
            y = os.path.join(dp, "merged_peninsula", "data.yaml")
            if os.path.isfile(y):
                n = yaml.safe_load(open(y, encoding="utf-8"))["names"]
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else n
    sys.exit("找不到 merged_peninsula/data.yaml")


class YoloDet:
    """YOLO txt 标注 -> torchvision 检测格式。类别 id 从 1 开始，0 留给背景。"""

    def __init__(self, paths, train):
        self.paths = paths
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as F
        ip = self.paths[i]
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        boxes, labels = [], []
        if os.path.isfile(lp):
            for line in open(lp, encoding="utf-8", errors="replace"):
                v = line.split()
                if len(v) < 5:
                    continue
                c = int(float(v[0]))
                cx, cy, w, h = (float(x) for x in v[1:5])
                x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
                if x1 <= x0 or y1 <= y0:
                    continue
                boxes.append([x0, y0, x1, y1])
                labels.append(c + 1)
        t = F.to_tensor(img)
        if self.train and random.random() < 0.5:
            t = torch.flip(t, dims=[2])
            boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64)}
        return t, target


def collate(b):
    return tuple(zip(*b))


def make_model(n_classes):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT")
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, n_classes + 1)
    return m


def evaluate(model, loader, dev):
    """返回 COCO 口径的 mAP@0.5，与 Ultralytics 的 mAP50 同定义。"""
    import torch
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False)
    model.eval()
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [i.to(dev) for i in imgs]
            out = model(imgs)
            metric.update([{k: v.cpu() for k, v in o.items()} for o in out],
                          [{k: v for k, v in t.items()} for t in tgts])
    r = metric.compute()
    return float(r["map_50"]) if "map_50" in r else float(r["map"])


def train_one(cfg_dir, seed, a, out_dir, n_classes):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = torch.utils.data.DataLoader(
        YoloDet(read_lines(os.path.join(cfg_dir, "train.txt")), True),
        batch_size=a.batch, shuffle=True, num_workers=a.workers,
        collate_fn=collate)
    va = torch.utils.data.DataLoader(
        YoloDet(read_lines(os.path.join(cfg_dir, "val.txt")), False),
        batch_size=a.batch, shuffle=False, num_workers=a.workers,
        collate_fn=collate)
    model = make_model(n_classes).to(dev)
    opt = torch.optim.SGD([p_ for p_ in model.parameters() if p_.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    os.makedirs(out_dir, exist_ok=True)
    best, best_ep, bad, ep = -1.0, -1, 0, 0
    for ep in range(a.epochs):
        model.train()
        for imgs, tgts in tr:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m50 = evaluate(model, va, dev)
        torch.save({"model": model.state_dict(), "epoch": ep,
                    "inner_map50": m50, "n_classes": n_classes},
                   os.path.join(out_dir, "last.pt"))
        if m50 > best:
            best, best_ep, bad = m50, ep, 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "inner_map50": m50, "n_classes": n_classes},
                       os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                break
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"    epoch {ep:>3}  inner mAP50 {m50:.4f}"
                  f"  best {best:.4f}@{best_ep}", flush=True)
    with open(os.path.join(out_dir, "epochs.txt"), "w") as fh:
        fh.write(f"{ep + 1}\n")
    return best, best_ep, ep + 1


def step_train(a, p):
    names = class_names(p["root"])
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isfile(os.path.join(p["clean_splits"], d,
                                                 "train.txt")))
    jobs = [(c, s) for s in a.seeds for c in cfgs]
    todo = [j for j in jobs if not os.path.isfile(os.path.join(
        p["runs"], f"clean_frcnn_{j[0]}_s{j[1]}", "weights", "best.pt"))]
    print(f"{ARCH}（torchvision, COCO 预训练）")
    print(f"{len(names)} 类: {names}")
    print(f"配置 {len(cfgs)}，种子 {a.seeds}，共 {len(jobs)} run，"
          f"待跑 {len(todo)}")
    print(f"epochs {a.epochs}  patience {a.patience}  batch {a.batch}  "
          f"lr {a.lr}")
    print("★ 与 YOLO 走同一套 clean_splits，只看内层验证集早停。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return
    t0 = time.time()
    for i, (cfg, seed) in enumerate(jobs, 1):
        name = f"clean_frcnn_{cfg}_s{seed}"
        out = os.path.join(p["runs"], name, "weights")
        if os.path.isfile(os.path.join(out, "best.pt")):
            continue
        print(f"\n[{i}/{len(jobs)}] {name}   已用 "
              f"{(time.time()-t0)/60:.0f} 分钟", flush=True)
        b, be, ne = train_one(os.path.join(p["clean_splits"], cfg), seed, a,
                              out, len(names))
        print(f"  完成：内层 best mAP50 {b:.4f} @ epoch {be}，共 {ne} 轮")
    print(f"\n结束，用时 {(time.time()-t0)/60:.1f} 分钟")


def step_eval(a, p):
    import torch
    names = class_names(p["root"])
    os.makedirs(p["results"], exist_ok=True)
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isfile(os.path.join(p["clean_splits"], d,
                                                 "train.txt")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for cfg in cfgs:
        outer = read_lines(os.path.join(p["splits"], cfg, "val.txt"))
        loader = torch.utils.data.DataLoader(
            YoloDet(outer, False), batch_size=a.batch, shuffle=False,
            num_workers=a.workers, collate_fn=collate)
        for seed in a.seeds:
            name = f"clean_frcnn_{cfg}_s{seed}"
            for tag in ("best", "last"):
                ck = os.path.join(p["runs"], name, "weights", f"{tag}.pt")
                if not os.path.isfile(ck):
                    continue
                st = torch.load(ck, map_location=dev, weights_only=False)
                model = make_model(st["n_classes"]).to(dev)
                model.load_state_dict(st["model"])
                m50 = evaluate(model, loader, dev)
                rows.append({"model": "frcnn-r50", "config": cfg,
                             "seed": seed, "checkpoint": tag,
                             "eval_on": "outer", "mAP50": m50})
                print(f"  {name} [{tag}]  mAP50 {m50:.4f}", flush=True)
    if rows:
        f = os.path.join(p["results"], "in_region_clean_frcnn.csv")
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n写出 {f}  ({len(rows)} 行)")
    print("\n★ 核对一件事再用这些数字：torchmetrics 的 map_50 与 Ultralytics "
          "的 mAP50 应当是同一口径。把某一折的 YOLO 权重同时用两套代码评一次，"
          "差值应在 0.01 以内；差得多就不要把两者放进同一张比较表。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True, choices=["train", "eval"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(os.path.abspath(a.root))
    {"train": step_train, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
