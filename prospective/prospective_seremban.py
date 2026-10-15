"""
Seremban 前瞻性测试（TASE 大修用）。用论文里已经冻结的 LOFO 模型（2026-09 训练、每个只见过 7 个半岛农场）
去测投稿后才采集的 Seremban 新农场。

三步，必须按顺序：
  1) prepare  把 Seremban 论文版 zip（Stretch 640）合并成一个 6 类数据集
        python prospective_seremban.py prepare --zips /root/autodl-tmp/durian/seremban_paper
  2) lock     记录：所有模型权重的 SHA-256 和时间、测试图片和标注的指纹、事先写下的预测和判定标准。
              生成 prospective_lock.json，并打印它的 SHA-256 —— 把这个哈希值发邮件给自己或 commit 到 GitHub，
              作为"打分之前就定好了"的时间戳证据。
        python prospective_seremban.py lock
  3) score    先核对锁定内容一个字节都没变，再逐个模型评估，按锁定的标准判定预测是否成立。
        python prospective_seremban.py score

默认路径都在 /root/autodl-tmp/durian 下，可以用 --root 改。
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import statistics as st
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

# ---------------- 论文里已经发表/投稿的数字（预测以这些为准，不看 Seremban 结果） ----------------
PAPER = {
    # 已对照 GitHub durian-site-split 的 results_clean/summary_main.csv 和 summary_per_unit.csv 核对
    "yolo11n": {"random_pooled": 0.428, "lofo": 0.223},
    "yolo11s": {"random_pooled": 0.462, "lofo": 0.214},
    "yolo11m": {"random_pooled": 0.465, "lofo": 0.233},
    "yolo11l": {"random_pooled": 0.472, "lofo": 0.236},
    "rtdetr-l": {"random_pooled": 0.499, "lofo": 0.221},
    "per_farm_range": [0.094, 0.356],          # 六个检测器平均后，单个半岛农场的范围
    "max_detector_gap": 0.022,                 # 检测器之间平均差距的上限
}
PRIMARY_FAMILY = "yolo11n"                       # 部署的模型
# 模型类别和顺序不写死：prepare 时从训练用的 merged_peninsula/data.yaml 读（和 clean_protocol_durian.py 同一个来源）
ALIASES = {  # Seremban 导出里的各种写法（统一小写下划线）→ 训练类别的小写写法
    "alga": "algal", "algal_spot": "algal",
    "leafhopper_damage": "leaf_hopper_damage",
}
EXCLUDED = {"scale_insect"}                      # 不属于论文六类，框会被去掉（图片保留）
IMG_EXTS = {".jpg", ".jpeg", ".png"}
RUN_RE = re.compile(r"clean_(?P<fam>.+?)_byfarm_fold(?P<fold>\d+)_s(?P<seed>\d+)$")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(n: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(n).strip().lower())


def stem_of(name: str) -> str:
    s = PurePosixPath(name).stem.split(".rf.")[0]
    return re.sub(r"_(jpe?g|png|heic|dng)$", "", s, flags=re.I)


def dhash(data: bytes) -> int:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("L").resize((9, 8))
        px = list(im.tobytes())
    v = 0
    for r in range(8):
        for c in range(8):
            v = (v << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return v


def training_classes(root: Path):
    import os
    import yaml
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("runs", "splits", "clean_splits", "results_v2",
                                               "results_clean", "prospective", "fb_runs", "fb2_runs",
                                               "nf_runs", "nf2_runs", "sc_runs")]
        if "merged_peninsula" in dns:
            y = Path(dp) / "merged_peninsula" / "data.yaml"
            if y.is_file():
                n = yaml.safe_load(y.read_text(encoding="utf-8"))["names"]
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n), str(y)
    sys.exit(f"在 {root} 下找不到 merged_peninsula/data.yaml")


def yaml_classes(y: Path):
    import yaml
    n = yaml.safe_load(y.read_text(encoding="utf-8"))["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)


def paths(root: Path):
    d = root / "prospective"
    return dict(dir=d, data=d / "seremban_640", yaml=d / "seremban_640" / "data.yaml",
                lock=d / "prospective_lock.json", prep=d / "prepare_report.txt",
                out_csv=d / "prospective_per_model.csv", out_txt=d / "prospective_result.txt")


# ============================================================ prepare
def cmd_prepare(args):
    P = paths(args.root)
    zdir = Path(args.zips)
    zips = sorted(zdir.rglob("*.zip"))
    if not zips:
        sys.exit(f"在 {zdir} 没找到 zip")
    if P["data"].exists() and any(P["data"].iterdir()):
        sys.exit(f"{P['data']} 已存在。prepare 只能跑一次；要重来请先删掉整个 {P['dir']}（锁定前才可以）。")
    import yaml
    MODEL_CLASSES, src_yaml = training_classes(args.root)
    idx = {norm(n): i for i, n in enumerate(MODEL_CLASSES)}
    print(f"训练类别（来自 {src_yaml}）：{MODEL_CLASSES}")
    recs, unknown, excluded = [], Counter(), Counter()

    def expand(label, zf):  # zip 里面套 zip（每个类别一个 Roboflow 导出）也能读
        inner = [n for n in zf.namelist() if n.lower().endswith(".zip") and "__MACOSX" not in n]
        if any(PurePosixPath(n).suffix.lower() in IMG_EXTS for n in zf.namelist()):
            yield label, zf
        for n in inner:
            yield from expand(f"{label}/{PurePosixPath(n).name}", zipfile.ZipFile(io.BytesIO(zf.read(n))))

    sources = [x for z in zips for x in expand(z.name, zipfile.ZipFile(z))]
    if not sources:
        sys.exit("zip 里没有找到任何图片（也没有内层 zip）。")
    for zname, zf in sources:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        tops = {n.split("/")[0] for n in names}
        pre = tops.pop() + "/" if len(tops) == 1 and all("/" in n for n in names) else ""
        rel = {n[len(pre):]: n for n in names}
        ymls = [r for r in rel if r.endswith((".yaml", ".yml")) and "/" not in r]
        cls = yaml.safe_load(zf.read(rel[ymls[0]]))["names"] if ymls else []
        if isinstance(cls, dict):
            cls = [cls[k] for k in sorted(cls, key=int)]
        n_img = 0
        for r, real in rel.items():
            p = PurePosixPath(r)
            if p.suffix.lower() not in IMG_EXTS or "images" not in p.parts:
                continue
            parts = list(p.parts)
            parts[parts.index("images")] = "labels"
            lbl = str(PurePosixPath(*parts).with_suffix(".txt"))
            boxes = []
            if lbl in rel:
                for row in zf.read(rel[lbl]).decode().splitlines():
                    v = row.split()
                    if not v:
                        continue
                    cname = cls[int(float(v[0]))] if cls else v[0]
                    key = norm(cname)
                    if key in EXCLUDED:
                        excluded[cname] += 1
                        continue
                    std = idx.get(ALIASES.get(key, key))
                    if std is None:
                        unknown[cname] += 1
                        continue
                    boxes.append((std, *map(float, v[1:5])))
            recs.append((stem_of(r), zf.read(real), boxes, zname))
            n_img += 1
        print(f"读取 {zname}：{n_img} 张，类别 {cls}")
    if unknown:
        sys.exit(f"有不认识的类别名，先在脚本 ALIASES 里补上再跑：{dict(unknown)}")

    # 同一张照片在多个类别项目里 → 合并（同名再用图片内容确认）
    by = defaultdict(list)
    for r in recs:
        by[r[0]].append(r)
    images = {}
    for stem, rs in by.items():
        clusters = []
        for s, data, boxes, zn in rs:
            h = dhash(data)
            for c in clusters:
                if bin(c["h"] ^ h).count("1") <= 6:
                    c["boxes"] += boxes
                    c["zips"].add(zn)
                    break
            else:
                clusters.append(dict(h=h, data=data, boxes=list(boxes), zips={zn}))
        for i, c in enumerate(clusters):
            images[stem + ("" if i == 0 else f"_v{i + 1}")] = c

    (P["data"] / "images" / "test").mkdir(parents=True)
    (P["data"] / "labels" / "test").mkdir(parents=True)
    per_cls = Counter()
    for stem, c in sorted(images.items()):
        (P["data"] / "images" / "test" / f"{stem}.jpg").write_bytes(c["data"])
        (P["data"] / "labels" / "test" / f"{stem}.txt").write_text(
            "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in c["boxes"]))
        for k in {b[0] for b in c["boxes"]}:
            per_cls[k] += 1
    P["yaml"].write_text(f"path: {P['data'].as_posix()}\ntrain: images/test\nval: images/test\n"
                         f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(MODEL_CLASSES)))
    lines = [f"Seremban 前瞻测试集  {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"zip：{[n for n, _ in sources]}",
             f"读入 {len(recs)} 条 → 合并后 {len(images)} 张不同的照片",
             f"去掉的非论文类别框：{dict(excluded)}",
             f"没有任何论文类别框的图（保留为负样本）：{sum(1 for c in images.values() if not c['boxes'])}"]
    lines += [f"  {MODEL_CLASSES[k]:<20} {per_cls[k]} 张图" for k in range(len(MODEL_CLASSES))]
    P["prep"].write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n数据在 {P['data']}。确认无误后运行 lock。")


# ============================================================ lock
def find_models(root: Path, ckpt: str):
    found = []
    for d in sorted((root / "runs").glob("clean_*_byfarm_fold*_s*")):
        m = RUN_RE.match(d.name)
        if not m or "frcnn" in m["fam"]:
            continue
        w = d / "weights" / f"{ckpt}.pt"
        if w.exists():
            found.append(dict(family=m["fam"], fold=int(m["fold"]), seed=int(m["seed"]), path=str(w)))
    return found


def dataset_fingerprint(data: Path):
    files = sorted((data / "images" / "test").iterdir()) + sorted((data / "labels" / "test").iterdir())
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    n_img = len(list((data / "images" / "test").iterdir()))
    return h.hexdigest(), n_img


def cmd_lock(args):
    P = paths(args.root)
    if P["lock"].exists():
        sys.exit(f"已经锁定过了：{P['lock']}。锁定只能做一次。")
    if not P["yaml"].exists():
        sys.exit("先运行 prepare。")
    print("锁定前请对照投稿稿 Table II 核对这些数字（锁定后不能改）：")
    for k, v in PAPER.items():
        print(f"  {k:<16} {v}")
    if input("和论文一致吗？输入 yes 继续：").strip().lower() != "yes":
        sys.exit("已取消。改脚本顶部的 PAPER 再运行。")
    models = find_models(args.root, args.ckpt)
    if not models:
        sys.exit(f"在 {args.root / 'runs'} 没找到 clean_*_byfarm_fold*_s* 的 {args.ckpt}.pt")
    print(f"找到 {len(models)} 个模型，计算哈希……")
    for m in models:
        p = Path(m["path"])
        m["sha256"] = sha256_file(p)
        m["mtime"] = dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
    fp, n_img = dataset_fingerprint(P["data"])
    MODEL_CLASSES = yaml_classes(P["yaml"])
    newest = max(m["mtime"] for m in models)
    fam_counts = Counter(m["family"] for m in models)
    mid = (PAPER[PRIMARY_FAMILY]["lofo"] + PAPER[PRIMARY_FAMILY]["random_pooled"]) / 2
    lock = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": "Prospective test of the farm-level protocol on a farm collected after submission (Seremban, 2026-10-11).",
        "checkpoint": args.ckpt,
        "models": models,
        "model_families": dict(fam_counts),
        "newest_model_mtime": newest,
        "test_set": {"path": str(P["data"]), "n_images": n_img, "fingerprint_sha256": fp,
                     "classes": MODEL_CLASSES, "excluded_classes": sorted(EXCLUDED),
                     "preprocessing": "Roboflow export, auto-orient, resize 640x640 (Stretch), as in the submitted paper"},
        "metric": "mAP50 over the classes present on the farm (Ultralytics val, imgsz 640, default conf/iou), "
                  "averaged over seeds then folds (each fold = one LOFO model set that never saw the farm)",
        "paper_values": PAPER,
        "predictions": {
            "P1_primary": f"{PRIMARY_FAMILY} mean mAP50 on Seremban < {mid:.4f} (midpoint of LOFO "
                          f"{PAPER[PRIMARY_FAMILY]['lofo']} and random-split {PAPER[PRIMARY_FAMILY]['random_pooled']}),"
                          " i.e. closer to the LOFO estimate than to the random-split estimate",
            "P2": f"{PRIMARY_FAMILY} mean mAP50 on Seremban lies within the per-farm range {PAPER['per_farm_range']}",
            "P3": "the five detector families' Seremban means differ by at most 0.05 mAP50 "
                  "(the farm, not the detector, sets the score)",
            "P4": "if both psyllid classes are present, at least one of them is among the two lowest-AP classes "
                  f"for {PRIMARY_FAMILY}",
            "P1_threshold": mid,
        },
        "notes": args.note or "",
    }
    P["lock"].write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    lh = sha256_file(P["lock"])
    print(f"\n模型：{dict(fam_counts)}；最新的模型文件时间：{newest}")
    print(f"测试集：{n_img} 张，指纹 {fp[:16]}…")
    print(f"预测 P1 门槛：{PRIMARY_FAMILY} < {mid:.4f}")
    print(f"\n已锁定：{P['lock']}")
    print(f"锁定文件 SHA-256：{lh}")
    print("👉 现在就把上面这个哈希值发邮件给自己（或 commit 到 GitHub），再运行 score。")


# ============================================================ score
def cmd_score(args):
    P = paths(args.root)
    if not P["lock"].exists():
        sys.exit("还没锁定。先运行 lock。")
    lock = json.loads(P["lock"].read_text(encoding="utf-8"))
    print(f"锁定文件 SHA-256：{sha256_file(P['lock'])}（应与你邮件里的一致）")
    fp, n_img = dataset_fingerprint(P["data"])
    if fp != lock["test_set"]["fingerprint_sha256"]:
        sys.exit("⚠ 测试集和锁定时不一样（图片或标注被改过）。停止。")
    for m in lock["models"]:
        if sha256_file(Path(m["path"])) != m["sha256"]:
            sys.exit(f"⚠ 模型被改过：{m['path']}。停止。")
    MODEL_CLASSES = lock["test_set"]["classes"]
    print(f"核对通过：{n_img} 张图、{len(lock['models'])} 个模型都和锁定时一致。\n")

    from ultralytics import YOLO, RTDETR
    rows = []
    for i, m in enumerate(lock["models"], 1):
        cls = RTDETR if "rtdetr" in m["family"] else YOLO
        model = cls(m["path"])
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if names != MODEL_CLASSES:
            sys.exit(f"⚠ 模型类别和预期不同：{m['path']} -> {names}")
        met = model.val(data=str(P["yaml"]), split="val", imgsz=640, batch=16, device=args.device,
                        plots=False, verbose=False, save_json=False, project=str(P["dir"] / "_val"),
                        name=f"{m['family']}_f{m['fold']}_s{m['seed']}", exist_ok=True)
        row = dict(family=m["family"], fold=m["fold"], seed=m["seed"], mAP50=float(met.box.map50),
                   mAP50_95=float(met.box.map))
        for j, c in enumerate(met.box.ap_class_index):
            row[f"AP50::{MODEL_CLASSES[int(c)]}"] = float(met.box.ap50[j])
        rows.append(row)
        print(f"[{i}/{len(lock['models'])}] {m['family']:<9} fold{m['fold']} s{m['seed']:<3} mAP50={row['mAP50']:.4f}")

    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k in ("family", "fold", "seed"), k))
    with open(P["out_csv"], "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # 先在 fold 内对 seed 平均，再对 fold 平均（和论文的汇总顺序一致）
    def fam_summary(fam):
        by_fold = defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by_fold[r["fold"]].append(r)
        fold_means = {f: st.mean(r["mAP50"] for r in rs) for f, rs in by_fold.items()}
        cls_keys = sorted({k for r in rows if r["family"] == fam for k in r if k.startswith("AP50::")})
        cls_means = {}
        for k in cls_keys:
            per_fold = [st.mean(r[k] for r in rs if k in r) for rs in by_fold.values() if any(k in r for r in rs)]
            cls_means[k[6:]] = st.mean(per_fold)
        return st.mean(fold_means.values()), fold_means, cls_means

    out = [f"Seremban 前瞻测试结果  {dt.datetime.now():%Y-%m-%d %H:%M}",
           f"锁定于 {lock['created']}；锁定文件 SHA-256 {sha256_file(P['lock'])}",
           f"测试集 {n_img} 张；模型 {len(rows)} 个（{lock['checkpoint']}.pt）", ""]
    fams = sorted({r["family"] for r in rows})
    fam_mean = {}
    for fam in fams:
        mean, folds, clsm = fam_summary(fam)
        fam_mean[fam] = mean
        ref = PAPER.get(fam, {})
        out.append(f"{fam:<9} Seremban mAP50 = {mean:.4f}   (论文 LOFO {ref.get('lofo', '-')}, random {ref.get('random_pooled', '-')})"
                   f"   fold 范围 {min(folds.values()):.3f}–{max(folds.values()):.3f}")
        if fam == PRIMARY_FAMILY:
            primary_cls = clsm
            for k, v in sorted(clsm.items(), key=lambda kv: kv[1]):
                out.append(f"      {k:<20} AP50 {v:.4f}")
    pred = lock["predictions"]
    p = fam_mean.get(PRIMARY_FAMILY)
    out.append("\n预测判定（按锁定时写下的标准）：")
    out.append(f"  P1 {PRIMARY_FAMILY} < {pred['P1_threshold']:.4f}：{'成立' if p < pred['P1_threshold'] else '不成立'}（{p:.4f}）")
    lo, hi = PAPER["per_farm_range"]
    out.append(f"  P2 在 {lo}–{hi} 之内：{'成立' if lo <= p <= hi else '不成立'}（{p:.4f}）")
    gap = max(fam_mean.values()) - min(fam_mean.values())
    out.append(f"  P3 检测器之间差距 ≤ 0.05：{'成立' if gap <= 0.05 else '不成立'}（{gap:.4f}）")
    ps = [k for k in primary_cls if norm(k) in ("psyllid", "psyllid_damage")]
    if len(ps) == 2:
        low2 = [k for k, _ in sorted(primary_cls.items(), key=lambda kv: kv[1])[:2]]
        ok = any(k in low2 for k in ps)
        out.append(f"  P4 木虱类在最低两类之中：{'成立' if ok else '不成立'}（最低两类 {low2}）")
    else:
        out.append(f"  P4 不适用（Seremban 上出现的木虱类：{ps}）")
    P["out_txt"].write_text("\n".join(out), encoding="utf-8")
    print("\n" + "\n".join(out))
    print(f"\n逐模型结果：{P['out_csv']}\n汇总：{P['out_txt']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "lock", "score"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/durian"))
        if name == "prepare":
            s.add_argument("--zips", required=True, help="Seremban 论文版（Stretch 640）zip 所在文件夹")
        if name == "lock":
            s.add_argument("--ckpt", default="best", choices=["best", "last"])
            s.add_argument("--note", default="", help="要一起锁定的说明，例如 QC 删图的规则")
        if name == "score":
            s.add_argument("--device", default="0")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "lock": cmd_lock, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
