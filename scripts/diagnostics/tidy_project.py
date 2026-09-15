#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tidy_project.py -- 把项目根目录整理成可提交的结构。

默认只报告不动手。看过计划再加 --apply。执行时会写出 TIDY_MANIFEST.csv
和 undo_tidy.py，任何一步都能原样退回。

整理后的结构：

    <root>/
      verify_claims.py            留在根，它按自身位置找 ./results_v2
      metadata.csv                留在根，多个脚本默认从根读
      attribution_by_content.csv  同上
      .phash_cache.csv            同上
      dataset_store/  results_v2/  runs/        原地不动
      scripts/
        prepare/      语料 -> 可分析数据集
        analyse/      训练、评估、聚合
        diagnostics/  审计与协变量
        legacy/       已被 v2 取代，保留以便审计
      outputs/        一次性产物 csv（可复算，不是输入）
      archive/
        zips/         压缩包
        results_v1/   旧结果目录
        scratch/      临时检查目录

用法：
    python tidy_project.py                # 只看计划
    python tidy_project.py --apply        # 执行
    python tidy_project.py --include-data # 连大数据目录一起归档（默认不动）
    python undo_tidy.py                   # 全部退回

不会删除任何东西，只移动。
"""

import argparse
import csv
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 留在根目录：脚本按相对路径依赖它们
KEEP_AT_ROOT = {
    "verify_claims.py",          # HERE/results_v2，移走就找不到表
    "metadata.csv",
    "metadata_backup.csv",
    "attribution_by_content.csv",
    ".phash_cache.csv",
    "requirements.txt",
    "README.md",
    "HANDOVER.md",
    "LICENSE",
    "manuscript.md",
    "manuscript.docx",
    "SELF_CHECK.md",
    "tidy_project.py",
    "undo_tidy.py",
    "TIDY_MANIFEST.csv",
}

# 目录：原地不动
KEEP_DIRS = {"dataset_store", "results_v2", "runs", "scripts", "outputs",
             "archive", "github", ".git", "figures", "docs"}

SCRIPTS = {
    "prepare": [
        "heic_to_jpg.py", "exif_audit.py", "merge_regions.py",
        "match_by_phash.py", "verify_phash.py", "audit_pool_join.py",
        "audit_duplicates.py", "rejoin_by_content.py", "make_splits_v2.py",
        "rebuild_bursts.py", "reconcile.py", "prepare_gwhd.py",
        "prepare_breakhis.py",
    ],
    "analyse": [
        "autodl_run.py", "colab_run_v2.py", "per_site_v2.py",
        "variance_decomposition.py", "site_count.py", "gwhd_run.py",
        "bkh_run.py", "aggregation_curve.py", "make_figures.py",
        "multiseed.py", "merge_cross.py", "q.py",
    ],
    "diagnostics": [
        "focal_lesion_check.py", "capture_conditions_v2.py",
        "audit_unmatched.py", "fix_shadowing.py", "abstention_v2.py",
        "verify_focal_block.py", "farm_covariates.py", "shortcut_check.py",
        "check_lesion_scale.py", "leafrot_ablation.py",
        "resolution_sweep.py", "audit_repo.py",
    ],
    "legacy": [
        "make_splits.py", "colab_run.py", "peninsula_by_burst.py",
        "sabah_by_tree.py", "abstention_analysis.py", "inspect_local.py",
        "check2.py", "diag.py", "pack_meta.py", "pack_v2.py",
        "pack_final.py", "pack_zenodo.py", "package_results.py",
        "rename_with_tree_id.py",
    ],
}

# 一次性产物：可以从脚本重算，不是输入
OUTPUTS = [
    "burst_assignment.csv", "reconcile_map.csv", "recovered_farms.csv",
    "recovered_farms_verified.csv", "pool_join_corrections.csv",
    "duplicate_farm_conflicts.csv", "AUDIT_REPORT.txt",
]

ARCHIVE = {
    "zips": [".zip", ".tar.gz", ".tgz", ".7z", ".rar"],
    "results_v1": ["results", "result v3", "results_v1_superseded"],
    "scratch": ["annotation_scale_check", "lesion_scale_check",
                "sabah_healthy_trunk", "second_dataset", "material"],
}

# 大数据目录：默认不动，--include-data 才归档
DATA_DIRS = ["durian_for_nature food", "freeze_peninsula_labelled_dataset",
             "freeze_sabah_lebeling_dataset", "freeze_sabah_original_dataset",
             "zenodo_upload"]


def plan(root, include_data):
    moves = []
    seen = set()

    def add(src, dst_dir, why):
        p = os.path.join(root, src)
        if not os.path.exists(p) or src in seen:
            return
        seen.add(src)
        moves.append((src, dst_dir, why))

    for kind, names in SCRIPTS.items():
        for n in names:
            add(n, f"scripts/{kind}", f"{kind} 脚本")

    for n in OUTPUTS:
        add(n, "outputs", "一次性产物，可重算")

    entries = sorted(os.listdir(root))
    for e in entries:
        if e in KEEP_AT_ROOT or e in KEEP_DIRS or e in seen:
            continue
        p = os.path.join(root, e)
        low = e.lower()
        if os.path.isfile(p):
            if any(low.endswith(x) for x in ARCHIVE["zips"]):
                add(e, "archive/zips", "压缩包")
        else:
            if e in ARCHIVE["results_v1"]:
                add(e, "archive/results_v1", "旧结果")
            elif e in ARCHIVE["scratch"]:
                add(e, "archive/scratch", "临时检查目录")
            elif e in DATA_DIRS and include_data:
                add(e, "archive/data", "原始数据（--include-data）")

    unclassified = []
    for e in entries:
        if e in KEEP_AT_ROOT or e in KEEP_DIRS or e in seen:
            continue
        if e in DATA_DIRS and not include_data:
            continue
        unclassified.append(e)
    return moves, unclassified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-data", action="store_true",
                    help="连原始数据目录一起归档（默认不动，它们很大）")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    if not os.path.isfile(os.path.join(root, "verify_claims.py")) and \
            not os.path.isdir(os.path.join(root, "results_v2")):
        print(f"{root} 看起来不是项目根目录（没有 verify_claims.py "
              f"也没有 results_v2/）")
        return 1

    moves, unc = plan(root, a.include_data)
    print(f"根目录: {root}")
    print(f"{'（预演，不会改动任何东西）' if not a.apply else '（执行中）'}\n")

    if moves:
        by_dst = {}
        for s, d, w in moves:
            by_dst.setdefault(d, []).append((s, w))
        for d in sorted(by_dst):
            print(f"  {d}/")
            for s, w in sorted(by_dst[d]):
                print(f"      {s:<38} {w}")
            print()
    else:
        print("  没有需要移动的东西。\n")

    if unc:
        print("未分类（保持原样，请自行确认）:")
        for e in unc:
            p = os.path.join(root, e)
            tag = "目录" if os.path.isdir(p) else \
                  f"{os.path.getsize(p)/1e6:.1f}MB"
            print(f"      {e:<38} {tag}")
        print()

    if not a.include_data:
        present = [d for d in DATA_DIRS
                   if os.path.isdir(os.path.join(root, d))]
        if present:
            print(f"原始数据目录保持原位（加 --include-data 才归档）:")
            for d in present:
                print(f"      {d}")
            print()

    if not a.apply:
        print("确认无误后加 --apply 执行。会写出 undo_tidy.py 可完全退回。")
        return 0

    done = []
    for s, d, w in moves:
        src = os.path.join(root, s)
        dst_dir = os.path.join(root, *d.split("/"))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(s))
        if os.path.exists(dst):
            print(f"  跳过 {s}：{d}/ 下已存在同名")
            continue
        try:
            shutil.move(src, dst)
            done.append((os.path.relpath(dst, root), s))
        except OSError as e:
            print(f"  ★ 移动失败 {s}: {e}")

    with open(os.path.join(root, "TIDY_MANIFEST.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_path", "old_path"])
        w.writerows(done)

    with open(os.path.join(root, "undo_tidy.py"), "w",
              encoding="utf-8") as fh:
        fh.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tidy_project.py --apply 的撤销脚本。把每个文件移回原位。"""
import csv, os, shutil, sys
root = os.path.dirname(os.path.abspath(__file__))
n = 0
with open(os.path.join(root, "TIDY_MANIFEST.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        src = os.path.join(root, r["new_path"])
        dst = os.path.join(root, r["old_path"])
        if not os.path.exists(src):
            print("缺失，跳过:", r["new_path"]); continue
        os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
        shutil.move(src, dst); n += 1
print(f"退回 {n} 项")
''')

    print(f"\n移动 {len(done)} 项")
    print(f"清单 TIDY_MANIFEST.csv，撤销 python undo_tidy.py")
    print("""
注意：脚本换了位置，运行方式变成

    python scripts/analyse/variance_decomposition.py --dir results_v2
    python scripts/diagnostics/focal_lesion_check.py

verify_claims.py 仍在根目录，直接 python verify_claims.py。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
