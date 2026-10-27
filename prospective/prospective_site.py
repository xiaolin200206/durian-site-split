"""
多站点前瞻性测试（Pahang / Penang / Malacca / Johor / Perak …），沿用 Seremban 的协议，
加了一道"出发前预锁定"（prelock），这样预测和模型哈希在采集之前就已经定下来了。

四步，必须按顺序，每个站点各跑一遍（--site 换站名）：

  1) prelock  出发前。记录 200 个 LOFO 模型的 SHA-256、PWA 预标模型 v0 的 SHA-256、
              事先写下的预测 P1–P4 和门槛、采集与标注协议说明。
              生成 prospective/<site>_prelock.json，打印它的 SHA-256 → 发邮件给自己 / commit。
        python prospective_site.py prelock --site pahang --pwa-weights /path/to/v0.pt --pwa-version "snapdurian-pwa 1.3.0" --note-file prospective/pahang/prelock_note.txt

  2) prepare  回来、QC 完标注之后。把导出的标注合成一个 6 类数据集 prospective/<site>_640/。
              来源可以是 Roboflow zip（和 Seremban 一样）或者一个普通的 YOLO 文件夹（PWA 导出）。
        python prospective_site.py prepare --site pahang --zips /path/to/zips
        python prospective_site.py prepare --site pahang --folder /path/to/pwa_export --resize640

  3) lock     核对 prelock 没变、模型没变，再加上测试集指纹，生成 prospective/<site>_lock.json，
              打印它的 SHA-256 → 再发一封邮件给自己 / commit。
        python prospective_site.py lock --site pahang --note "..."

  4) score    核对 prelock、lock、模型、测试集全部一致，再逐个模型评估，按锁定的标准判定。
        python prospective_site.py score --site pahang --device 0

默认根目录 /root/autodl-tmp/durian，可用 --root 改。Seremban 的 prospective_seremban.py 保持不动。
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

# ---------------- 论文里已经投稿的数字（预测以这些为准，和 Seremban 一字不改） ----------------
PAPER = {
    "yolo11n": {"random_pooled": 0.428, "lofo": 0.223},
    "yolo11s": {"random_pooled": 0.462, "lofo": 0.214},
    "yolo11m": {"random_pooled": 0.465, "lofo": 0.233},
    "yolo11l": {"random_pooled": 0.472, "lofo": 0.236},
    "rtdetr-l": {"random_pooled": 0.499, "lofo": 0.221},
    "per_farm_range": [0.094, 0.356],          # 六个检测器平均后，单个半岛农场的范围
    "max_detector_gap": 0.022,                 # 检测器之间平均差距的上限
}
PRIMARY_FAMILY = "yolo11n"                       # 部署的模型

# 各种写法 → 训练类别的小写写法（PWA 的 12 类名字也在这里）
ALIASES = {
    "alga": "algal", "algal_spot": "algal",
    "leafhopper_damage": "leaf_hopper_damage", "leaf_hopper": "leaf_hopper_damage",
    "psyllid_dmg": "psyllid_damage",
}
# 不属于论文六类，框会被去掉（图片保留为负样本）。
EXCLUDED = {"scale_insect", "root_disease", "pink_disease", "stem_borer", "weevil", "weevil_damage"}
# PWA 产品模型（11 类）里这两类同时覆盖论文的 Phomopsis 和 Leaf_rot，不能自动映射：
# QC 时每个框都要人工改成 Phomopsis 或 Leaf_rot。prepare 遇到还没改的就停下。
UNRESOLVED = {"leaf_spot_blight", "web_blight"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
RUN_RE = re.compile(r"clean_(?P<fam>.+?)_byfarm_fold(?P<fold>\d+)_s(?P<seed>\d+)$")
TREE_RE = re.compile(r"^[a-z]+\d*t(?P<tree>\d+)_", re.I)      # p1t3_IMG_0001 -> tree 3
SITE_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


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


def paths(root: Path, site: str):
    if not SITE_RE.match(site):
        sys.exit(f"站名只能用小写字母、数字和下划线：{site!r}")
    d = root / "prospective"
    return dict(dir=d, site=site, data=d / f"{site}_640", yaml=d / f"{site}_640" / "data.yaml",
                prelock=d / f"{site}_prelock.json", lock=d / f"{site}_lock.json",
                prep=d / f"{site}_prepare_report.txt",
                out_csv=d / f"{site}_per_model.csv", out_txt=d / f"{site}_result.txt")


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


def hash_models(models):
    for m in models:
        p = Path(m["path"])
        m["sha256"] = sha256_file(p)
        m["mtime"] = dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
    return models


def dataset_fingerprint(data: Path):
    files = sorted((data / "images" / "test").iterdir()) + sorted((data / "labels" / "test").iterdir())
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    n_img = len(list((data / "images" / "test").iterdir()))
    return h.hexdigest(), n_img


def predictions(site: str):
    mid = (PAPER[PRIMARY_FAMILY]["lofo"] + PAPER[PRIMARY_FAMILY]["random_pooled"]) / 2
    return {
        "P1_primary": f"{PRIMARY_FAMILY} mean mAP50 on {site} < {mid:.4f} (midpoint of LOFO "
                      f"{PAPER[PRIMARY_FAMILY]['lofo']} and random-split {PAPER[PRIMARY_FAMILY]['random_pooled']}),"
                      " i.e. closer to the LOFO estimate than to the random-split estimate",
        "P2": f"{PRIMARY_FAMILY} mean mAP50 on {site} lies within the per-farm range {PAPER['per_farm_range']}",
        "P3": f"the five detector families' {site} means differ by at most 0.05 mAP50 "
              "(the farm, not the detector, sets the score)",
        "P4": "if both psyllid classes are present, at least one of them is among the two lowest-AP classes "
              f"for {PRIMARY_FAMILY}",
        "P1_threshold": mid,
    }


# ============================================================ prelock（出发前）
def cmd_prelock(args):
    P = paths(args.root, args.site)
    if P["prelock"].exists():
        sys.exit(f"已经预锁定过了：{P['prelock']}。预锁定只能做一次；要重来就换个站名。")
    note = Path(args.note_file).read_text(encoding="utf-8").strip() if args.note_file else (args.note or "")
    if not note:
        sys.exit("预锁定必须带说明（--note 或 --note-file）：采集协议、预标模型、QC 规则。")
    print("预锁定前请对照投稿稿 Table 1 核对这些数字（锁定后不能改）：")
    for k, v in PAPER.items():
        print(f"  {k:<16} {v}")
    if not args.yes and input("和论文一致吗？输入 yes 继续：").strip().lower() != "yes":
        sys.exit("已取消。改脚本顶部的 PAPER 再运行。")
    if args.from_lock:
        # 不在 GPU 机器上时：直接沿用一份已经锁定、已经公开的模型哈希清单（例如 Seremban 的 prospective_lock.json）。
        # lock / score 时仍会逐个核对真实权重文件的哈希，对不上就停。
        src = json.loads(Path(args.from_lock).read_text(encoding="utf-8"))
        models = [dict(m) for m in src["models"]]
        MODEL_CLASSES = src["test_set"]["classes"] if "test_set" in src else src["classes"]
        src_yaml = f"from lock file {args.from_lock} (SHA-256 {sha256_file(Path(args.from_lock))})"
        print(f"从 {args.from_lock} 读入 {len(models)} 个 LOFO 模型的哈希")
    else:
        models = find_models(args.root, args.ckpt)
        if not models:
            sys.exit(f"在 {args.root / 'runs'} 没找到 clean_*_byfarm_fold*_s* 的 {args.ckpt}.pt")
        print(f"找到 {len(models)} 个 LOFO 模型，计算哈希……")
        hash_models(models)
        MODEL_CLASSES, src_yaml = training_classes(args.root)
    prelabel = None
    if args.pwa_weights:
        ws = [Path(w) for w in args.pwa_weights]
        for w in ws:
            if not w.exists():
                sys.exit(f"PWA 模型文件不存在：{w}")
        prelabel = {
            "role": "model that pre-labels images inside the PWA (NOT among the scored LOFO models)",
            "version": args.pwa_version or "",
            "files": [dict(path=args.pwa_path_as or str(w), sha256=sha256_file(w),
                           mtime=dt.datetime.fromtimestamp(w.stat().st_mtime).isoformat(timespec="seconds"))
                      for w in ws],
        }
    fam_counts = Counter(m["family"] for m in models)
    pre = {
        "stage": "prelock",
        "site": args.site,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": f"Prospective test of the farm-level protocol on a farm collected after submission ({args.site}); "
                   "predictions and model hashes fixed BEFORE the farm was photographed.",
        "checkpoint": args.ckpt,
        "models": models,
        "model_families": dict(fam_counts),
        "newest_model_mtime": max(m["mtime"] for m in models),
        "classes": MODEL_CLASSES,
        "classes_source": src_yaml,
        "excluded_classes": sorted(EXCLUDED),
        "tree_rule": args.tree_rule,
        "prelabel_model": prelabel,
        "metric": "mAP50 over the classes present on the farm (Ultralytics val, imgsz 640, default conf/iou), "
                  "averaged over seeds then folds (each fold = one LOFO model set that never saw the farm)",
        "paper_values": PAPER,
        "predictions": predictions(args.site),
        "protocol_note": note,
    }
    P["dir"].mkdir(parents=True, exist_ok=True)
    P["prelock"].write_text(json.dumps(pre, indent=2, ensure_ascii=False), encoding="utf-8")
    lh = sha256_file(P["prelock"])
    print(f"\n模型：{dict(fam_counts)}；最新的模型文件时间：{pre['newest_model_mtime']}")
    if prelabel:
        print(f"PWA 预标模型：{[f['sha256'][:12] + '…' for f in prelabel['files']]}  版本 {prelabel['version']}")
    print(f"预测 P1 门槛：{PRIMARY_FAMILY} < {pre['predictions']['P1_threshold']:.4f}")
    print(f"树的分配规则：{args.tree_rule}（prepare 只收这些树）")
    print(f"\n已预锁定：{P['prelock']}")
    print(f"预锁定文件 SHA-256：{lh}")
    print("👉 现在就把上面这个哈希值发邮件给自己（或 commit 到 GitHub）。然后才出发。")


# ============================================================ prepare（回来、QC 完之后）
def _read_sources_from_zips(zdir: Path):
    import yaml
    zips = sorted(zdir.rglob("*.zip"))
    if not zips:
        sys.exit(f"在 {zdir} 没找到 zip")

    def expand(label, zf):
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
        items = []
        for r, real in rel.items():
            p = PurePosixPath(r)
            if p.suffix.lower() not in IMG_EXTS or "images" not in p.parts:
                continue
            parts = list(p.parts)
            parts[parts.index("images")] = "labels"
            lbl = str(PurePosixPath(*parts).with_suffix(".txt"))
            rows = zf.read(rel[lbl]).decode().splitlines() if lbl in rel else []
            items.append((r, zf.read(real), rows))
        yield zname, cls, items


def _read_sources_from_folder(folder: Path):
    """普通 YOLO 文件夹：images/*.jpg + labels/*.txt，类别名来自 data.yaml / classes.txt，
    也接受 images/ 和 labels/ 下再分子文件夹（train/val/test 都会被读进来）。"""
    import yaml
    cls = []
    for y in (folder / "data.yaml", folder / "data.yml", folder / "dataset.yaml"):
        if y.is_file():
            n = yaml.safe_load(y.read_text(encoding="utf-8"))["names"]
            cls = [n[k] for k in sorted(n, key=int)] if isinstance(n, dict) else list(n)
            break
    if not cls and (folder / "classes.txt").is_file():
        cls = [l.strip() for l in (folder / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    img_dir, lbl_dir = folder / "images", folder / "labels"
    if not img_dir.is_dir():
        sys.exit(f"{folder} 下没有 images/ 文件夹")
    items = []
    for ip in sorted(img_dir.rglob("*")):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        rel = ip.relative_to(img_dir)
        lp = (lbl_dir / rel).with_suffix(".txt")
        rows = lp.read_text(encoding="utf-8").splitlines() if lp.is_file() else []
        items.append((str(rel), ip.read_bytes(), rows))
    if not items:
        sys.exit(f"{img_dir} 下没有图片")
    yield folder.name, cls, items


def _stretch640(data: bytes) -> bytes:
    """和 Roboflow 的 Stretch to 640x640 一致（标签是归一化坐标，不用改）。"""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((640, 640), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return buf.getvalue()


def cmd_prepare(args):
    P = paths(args.root, args.site)
    if not P["prelock"].exists():
        sys.exit(f"没有预锁定文件 {P['prelock']}。prepare 必须在 prelock 之后（证明预测先于数据）。")
    if P["data"].exists() and any(P["data"].iterdir()):
        sys.exit(f"{P['data']} 已存在。prepare 只能跑一次；要重来请先删掉 {P['data']}（lock 之前才可以）。")
    pre = json.loads(P["prelock"].read_text(encoding="utf-8"))
    MODEL_CLASSES = pre["classes"]
    idx = {norm(n): i for i, n in enumerate(MODEL_CLASSES)}
    print(f"训练类别（来自预锁定）：{MODEL_CLASSES}")
    sources = _read_sources_from_zips(Path(args.zips)) if args.zips else _read_sources_from_folder(Path(args.folder))
    tree_rule = pre.get("tree_rule", "all")
    recs, unknown, excluded, unresolved = [], Counter(), Counter(), Counter()
    no_tree, other_trees = [], Counter()
    for sname, cls, items in sources:
        n_img = 0
        for r, data, rows in items:
            if tree_rule != "all":
                m = TREE_RE.match(PurePosixPath(r).name)
                if not m:
                    no_tree.append(PurePosixPath(r).name)
                    continue
                t = int(m["tree"])
                if (tree_rule == "odd") != (t % 2 == 1):
                    other_trees[f"t{t}"] += 1
                    continue
            boxes = []
            for row in rows:
                v = row.split()
                if not v:
                    continue
                cname = cls[int(float(v[0]))] if cls else v[0]
                key = norm(cname)
                if key in UNRESOLVED:
                    unresolved[cname] += 1
                    continue
                if key in EXCLUDED:
                    excluded[cname] += 1
                    continue
                std = idx.get(ALIASES.get(key, key))
                if std is None:
                    unknown[cname] += 1
                    continue
                boxes.append((std, *map(float, v[1:5])))
            if args.resize640:
                data = _stretch640(data)
            recs.append((stem_of(r), data, boxes, sname))
            n_img += 1
        print(f"读取 {sname}：{n_img} 张，类别 {cls}")
    if no_tree:
        sys.exit(f"{len(no_tree)} 张图的文件名里没有树号（应该像 p1t3_IMG_0001.jpg），例如 {no_tree[:3]}。"
                 "先按树改名再跑。")
    if other_trees:
        print(f"按预锁定的规则（{tree_rule}）不属于这个测试集、已跳过：{dict(sorted(other_trees.items()))}")
    if unresolved:
        sys.exit(f"还有 {sum(unresolved.values())} 个没改的产品类框 {dict(unresolved)}。"
                 "这两类同时覆盖 Phomopsis 和 Leaf_rot，QC 时要逐个改成其中一类（按 2.2 节标准）再导出。")
    if unknown:
        sys.exit(f"有不认识的类别名，先在脚本 ALIASES 里补上再跑：{dict(unknown)}")

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
    per_cls, trees = Counter(), Counter()
    for stem, c in sorted(images.items()):
        (P["data"] / "images" / "test" / f"{stem}.jpg").write_bytes(c["data"])
        (P["data"] / "labels" / "test" / f"{stem}.txt").write_text(
            "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in c["boxes"]))
        for k in {b[0] for b in c["boxes"]}:
            per_cls[k] += 1
        trees[stem.split("_")[0]] += 1
    P["yaml"].write_text(f"path: {P['data'].as_posix()}\ntrain: images/test\nval: images/test\n"
                         f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(MODEL_CLASSES)))
    lines = [f"{args.site} 前瞻测试集  {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"来源：{args.zips or args.folder}" + ("（已 stretch 到 640×640）" if args.resize640 else ""),
             f"读入 {len(recs)} 条 → 合并后 {len(images)} 张不同的照片",
             f"去掉的非论文类别框：{dict(excluded)}",
             f"没有任何论文类别框的图（保留为负样本）：{sum(1 for c in images.values() if not c['boxes'])}",
             f"按文件名前缀分的树：{dict(sorted(trees.items()))}"]
    lines += [f"  {MODEL_CLASSES[k]:<20} {per_cls[k]} 张图" for k in range(len(MODEL_CLASSES))]
    P["prep"].write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n数据在 {P['data']}。确认无误（尤其树的分组和每类张数）后运行 lock。")


# ============================================================ lock（标注定稿后）
def cmd_lock(args):
    P = paths(args.root, args.site)
    if P["lock"].exists():
        sys.exit(f"已经锁定过了：{P['lock']}。锁定只能做一次。")
    if not P["prelock"].exists():
        sys.exit("没有预锁定文件。")
    if not P["yaml"].exists():
        sys.exit("先运行 prepare。")
    pre = json.loads(P["prelock"].read_text(encoding="utf-8"))
    pre_sha = sha256_file(P["prelock"])
    print(f"预锁定文件 SHA-256：{pre_sha}（应与你出发前邮件里的一致）")
    print("核对模型未被改动……")
    for m in pre["models"]:
        if sha256_file(Path(m["path"])) != m["sha256"]:
            sys.exit(f"⚠ 模型和预锁定时不一样：{m['path']}。停止。")
    if pre.get("prelabel_model"):
        for f in pre["prelabel_model"]["files"]:
            if Path(f["path"]).exists() and sha256_file(Path(f["path"])) != f["sha256"]:
                print(f"（提示）PWA 预标模型文件已变化：{f['path']}——如果你在 QC 后更新了 v1 这是正常的，"
                      f"prelock 里记录的是 v0 的哈希。")
    fp, n_img = dataset_fingerprint(P["data"])
    MODEL_CLASSES = yaml_classes(P["yaml"])
    if MODEL_CLASSES != pre["classes"]:
        sys.exit(f"⚠ 测试集类别 {MODEL_CLASSES} 和预锁定 {pre['classes']} 不一致。停止。")
    lock = {
        "stage": "lock",
        "site": args.site,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "prelock_file": str(P["prelock"]),
        "prelock_sha256": pre_sha,
        "prelock_created": pre["created"],
        "checkpoint": pre["checkpoint"],
        "models": pre["models"],
        "model_families": pre["model_families"],
        "newest_model_mtime": pre["newest_model_mtime"],
        "prelabel_model": pre.get("prelabel_model"),
        "test_set": {"path": str(P["data"]), "n_images": n_img, "fingerprint_sha256": fp,
                     "classes": MODEL_CLASSES, "excluded_classes": sorted(EXCLUDED),
                     "preprocessing": "auto-orient, resize 640x640 (stretch), as in the submitted paper"},
        "metric": pre["metric"],
        "paper_values": pre["paper_values"],
        "predictions": pre["predictions"],
        "protocol_note": pre["protocol_note"],
        "annotation_note": args.note or "",
    }
    P["lock"].write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    lh = sha256_file(P["lock"])
    print(f"测试集：{n_img} 张，指纹 {fp[:16]}…")
    print(f"\n已锁定：{P['lock']}")
    print(f"锁定文件 SHA-256：{lh}")
    print("👉 把这个哈希值也发邮件给自己（或 commit），再运行 score。")


# ============================================================ score
def cmd_score(args):
    P = paths(args.root, args.site)
    if not P["lock"].exists():
        sys.exit("还没锁定。先运行 lock。")
    lock = json.loads(P["lock"].read_text(encoding="utf-8"))
    print(f"锁定文件 SHA-256：{sha256_file(P['lock'])}（应与你邮件里的一致）")
    if sha256_file(P["prelock"]) != lock["prelock_sha256"]:
        sys.exit("⚠ 预锁定文件在锁定之后被改过。停止。")
    fp, n_img = dataset_fingerprint(P["data"])
    if fp != lock["test_set"]["fingerprint_sha256"]:
        sys.exit("⚠ 测试集和锁定时不一样（图片或标注被改过）。停止。")
    for m in lock["models"]:
        if sha256_file(Path(m["path"])) != m["sha256"]:
            sys.exit(f"⚠ 模型被改过：{m['path']}。停止。")
    MODEL_CLASSES = lock["test_set"]["classes"]
    print(f"核对通过：{n_img} 张图、{len(lock['models'])} 个模型、预锁定文件都和锁定时一致。\n")

    from ultralytics import YOLO, RTDETR
    rows = []
    for i, m in enumerate(lock["models"], 1):
        cls = RTDETR if "rtdetr" in m["family"] else YOLO
        model = cls(m["path"])
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if names != MODEL_CLASSES:
            sys.exit(f"⚠ 模型类别和预期不同：{m['path']} -> {names}")
        met = model.val(data=str(P["yaml"]), split="val", imgsz=640, batch=16, device=args.device,
                        plots=False, verbose=False, save_json=False, project=str(P["dir"] / f"_val_{args.site}"),
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

    site = args.site
    out = [f"{site} 前瞻测试结果  {dt.datetime.now():%Y-%m-%d %H:%M}",
           f"预锁定于 {lock['prelock_created']}（SHA-256 {lock['prelock_sha256']}）",
           f"锁定于 {lock['created']}；锁定文件 SHA-256 {sha256_file(P['lock'])}",
           f"测试集 {n_img} 张；模型 {len(rows)} 个（{lock['checkpoint']}.pt）", ""]
    fams = sorted({r["family"] for r in rows})
    fam_mean, primary_cls = {}, {}
    for fam in fams:
        mean, folds, clsm = fam_summary(fam)
        fam_mean[fam] = mean
        ref = PAPER.get(fam, {})
        out.append(f"{fam:<9} {site} mAP50 = {mean:.4f}   (论文 LOFO {ref.get('lofo', '-')}, random {ref.get('random_pooled', '-')})"
                   f"   fold 范围 {min(folds.values()):.3f}–{max(folds.values()):.3f}")
        if fam == PRIMARY_FAMILY:
            primary_cls = clsm
            for k, v in sorted(clsm.items(), key=lambda kv: kv[1]):
                out.append(f"      {k:<20} AP50 {v:.4f}")
    pred = lock["predictions"]
    p = fam_mean.get(PRIMARY_FAMILY)
    out.append("\n预测判定（按预锁定时写下的标准）：")
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
        out.append(f"  P4 不适用（{site} 上出现的木虱类：{ps}）")
    P["out_txt"].write_text("\n".join(out), encoding="utf-8")
    print("\n" + "\n".join(out))
    print(f"\n逐模型结果：{P['out_csv']}\n汇总：{P['out_txt']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prelock", "prepare", "lock", "score"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/durian"))
        s.add_argument("--site", required=True, help="站名，小写，如 pahang / penang / malacca / johor / perak")
        if name == "prelock":
            s.add_argument("--ckpt", default="best", choices=["best", "last"])
            s.add_argument("--pwa-weights", nargs="*", help="PWA 里预标用的 v0 权重文件（.pt / .onnx），可多个")
            s.add_argument("--pwa-version", default="", help="PWA 版本号或 build 号")
            s.add_argument("--note", default="", help="采集与标注协议说明（或用 --note-file）")
            s.add_argument("--note-file", default="", help="从文件读说明")
            s.add_argument("--yes", action="store_true", help="跳过 PAPER 数字的确认提示")
            s.add_argument("--tree-rule", default="all", choices=["all", "odd", "even"],
                           help="哪些树属于这个测试集（按文件名里的 tN）：all / odd 单数号 / even 双数号")
            s.add_argument("--from-lock", default="", help="沿用已锁定文件里的模型哈希（不在 GPU 机器上时用）")
            s.add_argument("--pwa-path-as", default="", help="在锁定文件里记录的 PWA 模型路径（GPU 机器上的路径）")
        if name == "prepare":
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--zips", help="Roboflow 导出 zip 所在文件夹（和 Seremban 一样）")
            g.add_argument("--folder", help="普通 YOLO 文件夹：images/ + labels/ + data.yaml 或 classes.txt")
            s.add_argument("--resize640", action="store_true", help="把图片 stretch 到 640×640（PWA 导出原图时用）")
        if name == "lock":
            s.add_argument("--note", default="", help="标注阶段的说明，例如 QC 修正率、从零标的子集数")
        if name == "score":
            s.add_argument("--device", default="0")
    args = ap.parse_args()
    {"prelock": cmd_prelock, "prepare": cmd_prepare, "lock": cmd_lock, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
