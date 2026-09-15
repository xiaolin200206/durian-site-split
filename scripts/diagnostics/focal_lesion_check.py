#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
focal_lesion_check.py -- 92.8 倍的病斑尺寸差，是不是被镜头焦距混淆出来的？

背景: reconcile.py 显示 2.22mm(超广角) 出现在全部五个分析农场，占比从
farm 5 的 2% 到 farm 6 的 69% 不等。而 92.8 倍正是 farm 6 比 farm 2，
恰好是 2.22mm 占比最高的农场比最低的农场。方向完全对得上一个假效应。

本脚本做三件事:
  [A] 用论文口径复算基线（每 class×farm 取所有框面积的中位数，
      图片数 >=5 才出数），确认能对上 farm_annotation_scale.csv。
  [B] 按焦距分层重算。若在单一焦距内 92.8 倍依然成立，混淆被排除。
  [C] 农场内对照 —— 同一个农场里 2.22mm 的框和 6.76mm 的框比。
      这是最干净的检验，因为农场当自己的对照，病害阶段、品种、
      管理全部固定，唯一变的是镜头。farm 3 (29 vs 30) 几乎平衡，
      是最好的检验点。

纯标准库。只读。

用法:
    python focal_lesion_check.py
    python focal_lesion_check.py --class-name Leaf_rot --imgsz 640
"""

import argparse
import csv
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

MIN_IMAGES = 5          # 与 farm_covariates.py 一致
PAPER_FARMS = ["0", "2", "3", "5", "6"]   # 运行时会按池中实际农场覆盖


def norm_farm(x):
    """'0.0' 与 '0' 统一。"""
    x = (x or "").strip()
    if not x:
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def norm_stem(x):
    return Path(str(x)).stem.lower()


def load_meta(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def focal_bin(v):
    """焦距归箱到 2 位小数。空值归 '?'。"""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "?"


def read_names(data_yaml):
    """不依赖 PyYAML，解析 names。覆盖三种写法：
       块状列表(yaml.safe_dump 的默认)、行内列表、字典。"""
    if not data_yaml.is_file():
        return None
    txt = data_yaml.read_text(encoding="utf-8", errors="replace")

    # 1) 块状列表:  names:\n- Algal\n- Leaf_rot
    m = re.search(r"^names\s*:\s*$", txt, re.M)
    if m:
        out = []
        for line in txt[m.end():].splitlines():
            if not line.strip():
                continue
            item = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if item:
                out.append(item.group(1).strip().strip("'\""))
                continue
            kv = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if kv:
                out.append(kv.group(2).strip().strip("'\""))
                continue
            break          # 遇到下一个顶层键，结束
        if out:
            return out

    # 2) 行内列表:  names: [Algal, Leaf_rot]
    m = re.search(r"^names\s*:\s*\[(.*?)\]", txt, re.M | re.S)
    if m:
        return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]

    # 3) 字典:  names:\n  0: Algal
    pairs = re.findall(r"^\s+(\d+)\s*:\s*(.+?)\s*$", txt, re.M)
    if pairs:
        return [n.strip().strip("'\"")
                for _, n in sorted(pairs, key=lambda x: int(x[0]))]
    return None


def read_boxes(label_dir):
    """stem -> [(cls_idx, normalised_area), ...]"""
    out = {}
    for p in label_dir.glob("*.txt"):
        rows = []
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                v = line.split()
                if len(v) >= 5:
                    try:
                        rows.append((int(float(v[0])), float(v[3]) * float(v[4])))
                    except ValueError:
                        pass
        except OSError:
            continue
        out[norm_stem(p.name)] = rows
    return out


def px(area, imgsz):
    return math.sqrt(area) * imgsz if area and area > 0 else float("nan")


def med(xs):
    return st.median(xs) if xs else None


def cell(areas, n_img, imgsz, gate=MIN_IMAGES):
    if n_img < gate or not areas:
        return "—"
    return f"{px(med(areas), imgsz):.0f}px"


def ratio_line(tag, a, b, na, nb, imgsz):
    """a,b 是两组框面积列表。打印面积倍数。"""
    if not a or not b:
        return f"    {tag:<34} 数据不足 (n_box {len(a)}/{len(b)})"
    ma, mb = med(a), med(b)
    hi, lo = (ma, mb) if ma >= mb else (mb, ma)
    r = hi / lo if lo > 0 else float("inf")
    warn = ""
    if min(na, nb) < MIN_IMAGES:
        warn = f"  ★ 单元过薄 (图片数 {na}/{nb})"
    return (f"    {tag:<34} {px(ma, imgsz):5.0f}px vs {px(mb, imgsz):5.0f}px"
            f"   面积 {r:6.1f}×   (框 {len(a)}/{len(b)}, 图 {na}/{nb}){warn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--attribution", default="attribution_by_content.csv",
                    help="按内容重建的农场归属；存在时优先于 metadata 的 farm 列")
    ap.add_argument("--use-metadata-farm", action="store_true",
                    help="强制退回旧的 metadata.farm（仅用于复现旧结果）")
    ap.add_argument("--farms", default=None,
                    help="要报告的农场，逗号分隔；默认取池中全部")
    ap.add_argument("--merged", default="dataset_store/merged_peninsula")
    ap.add_argument("--class-name", default="Leaf_rot")
    ap.add_argument("--class-index", type=int, default=None,
                    help="直接给类别索引，绕过 data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--emit", default=None,
                    help="把 farm x focal 的分层结果写成 CSV，供 verify_claims 使用")
    ap.add_argument("--all-classes", action="store_true",
                    help="遍历全部类别（配合 --emit 一次产出完整表）")
    ap.add_argument("--scores", default="results_v2/in_region.csv",
                    help="新的折分数表；给空串则跳过 stage 关联")
    ap.add_argument("--split-assignment",
                    default="dataset_store/splits_B/split_assignment.csv")
    args = ap.parse_args()

    root = Path(args.root)
    meta_p = root / args.meta
    merged = root.joinpath(*re.split(r"[\\/]", args.merged))
    if not meta_p.is_file():
        print(f"找不到 {meta_p}")
        return 1
    label_dir = merged / "labels"
    if not label_dir.is_dir():
        print(f"找不到 {label_dir}")
        return 1

    names = read_names(merged / "data.yaml")
    boxes = read_boxes(label_dir)

    if not names:
        seen = sorted({c for bs in boxes.values() for c, _ in bs})
        names = [f"class_{i}" for i in range(max(seen) + 1)] if seen else []
        print("!! 无法从 data.yaml 读出类别名，改用标注里的索引。")
        print(f"   data.yaml 前几行: "
              f"{(merged / 'data.yaml').read_text(encoding='utf-8', errors='replace')[:200]!r}"
              if (merged / "data.yaml").is_file() else "   (data.yaml 不存在)")
        print(f"   标注里出现的类别索引: {seen}")
        print("   可用 --class-index 直接指定，例如 --class-index 1")

    print("=" * 78)
    print(f"类别表: {names}")
    meta = {norm_stem(r.get("stem") or r.get("file") or ""): r for r in load_meta(meta_p)}
    print(f"标注文件 {len(boxes)} 个 | metadata {len(meta)} 行")

    # ---- 农场归属：优先用按内容重建的结果 ----
    attr_p = root / args.attribution
    content_farm, matched_of = {}, {}
    src_label = "metadata.farm（旧的按文件名 join）"
    if attr_p.is_file() and not args.use_metadata_farm:
        with open(attr_p, newline="", encoding="utf-8-sig",
                  errors="replace") as fh:
            for r in csv.DictReader(fh):
                if (r.get("status") or "").strip() == "ok":
                    f = (r.get("farm_new") or "").strip()
                    if f:
                        s = norm_stem(r.get("stem") or "")
                        content_farm[s] = f
                        matched_of[s] = norm_stem(r.get("matched_file") or "")
        src_label = f"{attr_p.name}（按内容匹配，{len(content_farm)} 张）"
    print(f"农场归属来源: {src_label}")

    def meta_for(stem):
        """标注图可能被平台改名（00006 这种），metadata 里没有该 stem。
        此时经 matched_file 指向的原图取 EXIF —— 那才是有元数据的记录。"""
        m = meta.get(stem)
        if m is not None:
            return m
        return meta.get(matched_of.get(stem, ""))

    pool = []
    via_self = via_matched = 0
    for stem, bs in boxes.items():
        if content_farm:
            farm = content_farm.get(stem, "")
            m = meta_for(stem) if farm else None
            if farm and m is not None:
                via_self += 1 if stem in meta else 0
                via_matched += 0 if stem in meta else 1
        else:
            m = meta.get(stem)
            farm = norm_farm((m or {}).get("farm", "")) if m else ""
        if not farm:
            continue
        pool.append((stem, farm, focal_bin((m or {}).get("FocalLength")),
                     ((m or {}).get("Model") or "<空>"), bs))
    if content_farm:
        print(f"元数据来源: 同名 {via_self} | 经 matched_file {via_matched}")
    print(f"分析池: {len(pool)} 张   (旧版 560, 新版应为 827)")

    if args.farms:
        PAPER_FARMS[:] = [f.strip() for f in args.farms.split(",")]
    else:
        PAPER_FARMS[:] = sorted({f for _, f, _, _, _ in pool},
                                key=lambda x: (len(x), x))
    print(f"报告的农场: {PAPER_FARMS}")
    print("=" * 78)

    if args.class_index is not None:
        cidx = args.class_index
        cname = names[cidx] if 0 <= cidx < len(names) else f"class_{cidx}"
    else:
        try:
            cidx = names.index(args.class_name)
            cname = args.class_name
        except ValueError:
            print(f"!! 类别 {args.class_name} 不在 {names}")
            print("   用 --class-index 指定索引，或检查 data.yaml")
            return 1
    args.class_name = cname

    # ---------------- [A] 基线复算 ----------------
    print(f"\n[A] 基线复算  class = {args.class_name}   (论文口径: 全框中位数, 图片数>=5)")
    areas_f = defaultdict(list)      # farm -> areas
    imgs_f = defaultdict(int)
    for stem, farm, foc, model, bs in pool:
        a = [ar for c, ar in bs if c == cidx]
        if a:
            areas_f[farm].extend(a)
            imgs_f[farm] += 1

    print(f"    {'farm':<8}{'图片数':>8}{'框数':>8}{'中位面积':>14}{'≈边长':>10}")
    vals = {}
    for f in PAPER_FARMS:
        a, n = areas_f.get(f, []), imgs_f.get(f, 0)
        if n >= MIN_IMAGES and a:
            vals[f] = med(a)
            print(f"    {f:<8}{n:>8}{len(a):>8}{med(a):>14.8f}{px(med(a), args.imgsz):>9.0f}px")
        else:
            print(f"    {f:<8}{n:>8}{len(a):>8}{'—':>14}{'—':>10}")
    if len(vals) >= 2:
        hi, lo = max(vals.values()), min(vals.values())
        fh = max(vals, key=vals.get); fl = min(vals, key=vals.get)
        print(f"\n    极值比: farm {fh} / farm {fl} = {hi/lo:.1f}×"
              f"      论文报的是 92.8×")

    # ---------------- 焦距分布 ----------------
    print(f"\n[B] 该类别图片的焦距分布（不是全部图片，只看含 {args.class_name} 的）")
    focs = sorted({f for _, _, f, _, _ in pool})
    grid = defaultdict(lambda: defaultdict(int))
    for stem, farm, foc, model, bs in pool:
        if any(c == cidx for c, _ in bs):
            grid[farm][foc] += 1
    print(f"    {'farm':<8}" + "".join(f"{f+'mm':>10}" for f in focs) + f"{'合计':>8}")
    for f in PAPER_FARMS:
        row = grid.get(f, {})
        print(f"    {f:<8}" + "".join(f"{row.get(fo, 0) or '.':>10}" for fo in focs)
              + f"{sum(row.values()):>8}")
    print("    ↑ 若 farm 6 高度集中在 2.22mm 而 farm 2 集中在 6.76mm，混淆方向成立")

    # ---------------- 分层重算 ----------------
    print(f"\n[C] 按焦距分层，farm × focal 的中位边长（'—' = 该单元图片数 <{MIN_IMAGES}）")
    ar_ff = defaultdict(list)
    im_ff = defaultdict(int)
    for stem, farm, foc, model, bs in pool:
        a = [x for c, x in bs if c == cidx]
        if a:
            ar_ff[(farm, foc)].extend(a)
            im_ff[(farm, foc)] += 1
    print(f"    {'farm':<8}" + "".join(f"{f+'mm':>12}" for f in focs))
    for f in PAPER_FARMS:
        cells = "".join(
            f"{cell(ar_ff.get((f, fo), []), im_ff.get((f, fo), 0), args.imgsz):>12}"
            for fo in focs)
        print(f"    {f:<8}{cells}")

    print(f"\n    单一焦距内的极值比（混淆若被排除，倍数应仍然很大）:")
    for fo in focs:
        cand = {f: med(ar_ff[(f, fo)]) for f in PAPER_FARMS
                if im_ff.get((f, fo), 0) >= MIN_IMAGES and ar_ff.get((f, fo))}
        if len(cand) >= 2:
            fh = max(cand, key=cand.get); fl = min(cand, key=cand.get)
            print(ratio_line(f"{fo}mm 内: farm {fh} / farm {fl}",
                             ar_ff[(fh, fo)], ar_ff[(fl, fo)],
                             im_ff[(fh, fo)], im_ff[(fl, fo)], args.imgsz))
        else:
            print(f"    {fo+'mm 内':<34} 可比农场不足 ({len(cand)} 个达标)")

    # ---------------- 农场内对照 ----------------
    print(f"\n[D] ★ 农场内对照 —— 同一农场，2.22mm 的框 vs 6.76mm 的框")
    print("    这是最干净的检验: 病害阶段/品种/管理全部固定，只有镜头变。")
    any_ok = False
    for f in PAPER_FARMS:
        a1, n1 = ar_ff.get((f, "2.22"), []), im_ff.get((f, "2.22"), 0)
        a2, n2 = ar_ff.get((f, "6.76"), []), im_ff.get((f, "6.76"), 0)
        if a1 and a2:
            any_ok = True
            print(ratio_line(f"farm {f}", a1, a2, n1, n2, args.imgsz))
        else:
            print(f"    {'farm ' + f:<34} 该农场缺一侧 (2.22: {n1} 图, 6.76: {n2} 图)")
    if any_ok:
        print("\n    判读: 若农场内两个焦距的中位边长接近（比如 <1.5×），"
              "\n          镜头不是驱动因素，92.8× 站得住。"
              "\n          若农场内就差出好几倍，92.8× 里有相当部分是镜头造的，"
              "\n          正文那条必须重写。")
    else:
        print("\n    没有农场同时具备两个焦距的足够样本 —— 这本身就是结论:"
              "\n    焦距和农场几乎共线，两者无法分离，正文必须如实说明。")

    # ---------------- 全池相关 ----------------
    print(f"\n[E] 全池层面: 焦距 -> 框面积（不分农场，看总体方向）")
    print(f"    {'focal':<10}{'图片数':>8}{'框数':>8}{'中位面积':>14}{'≈边长':>10}")
    ar_f = defaultdict(list); im_f = defaultdict(int)
    for stem, farm, foc, model, bs in pool:
        a = [x for c, x in bs if c == cidx]
        if a:
            ar_f[foc].extend(a); im_f[foc] += 1
    for fo in focs:
        if im_f.get(fo):
            print(f"    {fo+'mm':<10}{im_f[fo]:>8}{len(ar_f[fo]):>8}"
                  f"{med(ar_f[fo]):>14.8f}{px(med(ar_f[fo]), args.imgsz):>9.0f}px")

    # ---------------- [F] stage 关联 ----------------
    if args.scores:
        print(f"\n[F] ★ stage 关联：病斑尺寸 vs 折分数")
        sp = root.joinpath(*re.split(r"[\\/]", args.scores))
        sap = root.joinpath(*re.split(r"[\\/]", args.split_assignment))
        if not sp.is_file() or not sap.is_file():
            print(f"    找不到 {sp} 或 {sap}，跳过")
        else:
            # fold -> farm
            fold_farm = {}
            with open(sap, newline="", encoding="utf-8-sig",
                      errors="replace") as fh:
                for r in csv.DictReader(fh):
                    for k, v in r.items():
                        if (k or "").startswith("split_farm_fold") and \
                                (v or "").strip().lower() == "val":
                            fold_farm[k.replace("split_farm_", "byfarm_")] = \
                                norm_farm(r.get("farm"))
            # config -> mean mAP50
            acc = defaultdict(list)
            with open(sp, newline="", encoding="utf-8-sig",
                      errors="replace") as fh:
                for r in csv.DictReader(fh):
                    c = (r.get("config") or "").strip()
                    if c.startswith("byfarm"):
                        try:
                            acc[c].append(float(r["mAP50"]))
                        except (KeyError, ValueError):
                            pass
            pairs = []
            print(f"    {'farm':<8}{'折分数':>9}{'中位边长':>10}{'图片数':>8}")
            for cfgn, vals in sorted(acc.items()):
                f = fold_farm.get(cfgn)
                if not f:
                    continue
                score = sum(vals) / len(vals)
                a, n = areas_f.get(f, []), imgs_f.get(f, 0)
                if n >= MIN_IMAGES and a:
                    side = px(med(a), args.imgsz)
                    pairs.append((f, score, side, n))
                    print(f"    {f:<8}{score:>9.4f}{side:>9.0f}px{n:>8}")
                else:
                    print(f"    {f:<8}{score:>9.4f}{'—':>10}{n:>8}"
                          f"   ({args.class_name} 图片数 <{MIN_IMAGES})")
            if len(pairs) >= 3:
                xs = [p[2] for p in pairs]
                ys = [p[1] for p in pairs]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                sxx = sum((x - mx) ** 2 for x in xs)
                syy = sum((y - my) ** 2 for y in ys)
                r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else 0.0
                print(f"\n    Pearson r = {r:+.3f}   (n={len(pairs)}，"
                      f"旧版 r=0.54 于 n=5)")
                print("""
    判读:
      r 仍为正且不小  -> 病斑越大分数越高，即病斑小的农场吃亏，
                        论文的食物系统论点成立
      r 转负或接近零  -> 该论点不成立，必须撤回。
                        新的最差折是 farm 6，而 farm 6 病斑最大。""")
            else:
                print(f"\n    可用农场不足（{len(pairs)} 个），无法计算")

    print("\n" + "=" * 78)

    # ---------------- 导出供 verify_claims 使用 ----------------
    if args.emit:
        out = Path(args.emit)
        if not out.is_absolute():
            out = root / args.emit
        out.parent.mkdir(parents=True, exist_ok=True)
        targets = range(len(names)) if args.all_classes else [cidx]
        recs = []
        for ci_ in targets:
            cn = names[ci_]
            a_ff, i_ff = defaultdict(list), defaultdict(int)
            a_f, i_f = defaultdict(list), defaultdict(int)
            for stem, farm, foc, model, bs in pool:
                a = [x for c, x in bs if c == ci_]
                if not a:
                    continue
                a_ff[(farm, foc)].extend(a); i_ff[(farm, foc)] += 1
                a_f[farm].extend(a); i_f[farm] += 1
            for farm in sorted(set(f for _, f, _, _, _ in pool)):
                # 未分层
                if i_f.get(farm, 0):
                    recs.append({"class": cn, "farm": farm, "focal": "ALL",
                                 "n_images": i_f[farm],
                                 "n_boxes": len(a_f[farm]),
                                 "median_area": med(a_f[farm]),
                                 "side_px": px(med(a_f[farm]), args.imgsz),
                                 "gated": i_f[farm] >= MIN_IMAGES})
                for fo in focs:
                    k = (farm, fo)
                    if i_ff.get(k, 0):
                        recs.append({"class": cn, "farm": farm, "focal": fo,
                                     "n_images": i_ff[k],
                                     "n_boxes": len(a_ff[k]),
                                     "median_area": med(a_ff[k]),
                                     "side_px": px(med(a_ff[k]), args.imgsz),
                                     "gated": i_ff[k] >= MIN_IMAGES})
        import csv as _csv
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["class", "farm", "focal",
                                                "n_images", "n_boxes",
                                                "median_area", "side_px",
                                                "gated"])
            w.writeheader()
            w.writerows(recs)
        print(f"已导出 {len(recs)} 行 -> {out}")
        print("verify_claims 可据此复算分层后的农场间倍数。")

    print("把 [D] 那一段发回来，那是决定 92.8× 要不要重写的关键。")
    print("换类别看:  python focal_lesion_check.py --class-name Phomopsis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
