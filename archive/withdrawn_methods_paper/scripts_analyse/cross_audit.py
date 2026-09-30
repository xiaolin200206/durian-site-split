#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_audit.py -- 正文、补充材料、图注、结果表之间的机器交叉审计。

verify_claims.py 回答的是「这个数字能不能从表里重算出来」。它抓不到另一类
错误：**同一个量在两处写得不一样**，或者**稿子关于自身设计的陈述前后矛盾**。
到目前为止这一类全部是外部读者发现的，不是检查发现的：

  - Methods 说排除十二棵 Sabah 树，Fig 1 图注说显示十二棵（Methods 写反了）
  - Methods 重抽 4,000 次，Fig 3 图注写 2,000
  - Table 1 写 GWHD 59 runs，正文却称 fully crossed
  - 正文引用的 Supplementary Table 9–12 一度根本不存在

这个脚本把这一类做成机器检查。它不判断对错，它列出**需要人看一眼的地方**，
所以输出里有噪声是正常的；目标是让真正的矛盾不可能藏在噪声之外。

用法：
    python cross_audit.py --root .
    python cross_audit.py --root . --section numbers
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# 同一个量在多处出现时使用的关键词。key 是概念，value 是抓取正则。
# 每个概念在全部文档里抽到的数值集合，若多于一个，就报出来人工判断。
CONCEPTS = {
    "durian farms": r"(\w+|\d+)\s+(?:peninsular\s+)?farms\b",
    "Sabah trees": r"(\w+|\d+)\s+Sabah trees",
    "GWHD sessions": r"(\w+|\d+)\s+GWHD sessions",
    "BreaKHis patients": r"(\w+|\d+)\s+BreaKHis patients",
    "HAR subjects": r"(\w+|\d+)\s+HAR subjects",
    "durian bursts": r"(\w+|\d+)\s+(?:peninsular\s+)?(?:durian\s+)?capture bursts",
    "resampling draws": r"([\d,]+)\s+(?:times|draws|resamples)",
    "runs total": r"([\d,]+)\s+(?:training\s+)?runs\b",
    "re-validations": r"([\d,]+)\s+re-validations",
    "checkpoint combinations": r"(\w+)\s+of the ten combinations",
    "heterogeneity share": r"(\d+)[–-]\d+% of (?:that|the observed) spread",
    "unit component": r"(\d+)[–-]\d+% of the variance",
    "model pairs resolvable": r"(\w+)\s+of eighteen pairs",
    "overstatement range": r"by ([\d.]+)[–-][\d.]+%",
    "d range": r"\*d\* = ([\d.]+) to [\d.]+",
}

WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "sixteen": 16,
            "twenty-four": 24, "twenty-six": 26, "thirty-two": 32,
            "forty-seven": 47, "fifty-eight": 58, "eighty-one": 81}


def tonum(s):
    s = s.strip().lower().replace(",", "")
    if s in WORD2NUM:
        return WORD2NUM[s]
    try:
        return float(s)
    except ValueError:
        return None


def split_sections(text):
    """按二级标题切块，便于报告位置。"""
    out, cur, buf = [], "(head)", []
    for line in text.split("\n"):
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            if buf:
                out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1)[:60], []
        else:
            buf.append(line)
    if buf:
        out.append((cur, "\n".join(buf)))
    return out


def check_concepts(docs, baseline, accept, baseline_path):
    """同一概念在多处的数值是否一致。

    这一节必然有误报——「用第二台相机的三个农场」和「八个农场」都会被抓到。
    所以用 linter 的办法：审一次，--accept 写入基线，以后只报**新出现**的
    组合。这样噪声只需要看一次，而真正的新分歧不会淹没在里面。
    """
    print("=" * 74)
    print("[1] 同一个量在多处出现，数值是否一致")
    print("=" * 74)
    issues = 0
    fresh = {}
    for concept, pat in CONCEPTS.items():
        seen = defaultdict(list)
        for docname, text in docs.items():
            for sect, body in split_sections(text):
                for m in re.finditer(pat, body, flags=re.I):
                    v = tonum(m.group(1))
                    if v is None:
                        continue
                    ctx = body[max(0, m.start() - 55):m.end() + 25]
                    ctx = " ".join(ctx.split())
                    seen[v].append(f"{docname}/{sect}: …{ctx}…")
        values = sorted(seen)
        fresh[concept] = values
        known = baseline.get(concept)
        if len(values) <= 1:
            continue
        if known is not None and values == known:
            continue          # 已经人工审过的组合
        issues += 1
        tag = "新增" if known is not None else "未审"
        print(f"\n  ★ {concept}: {len(values)} 个不同数值（{tag}）")
        new_vals = ([v for v in values if v not in known] if known else values)
        for v in values:
            mark = "→" if v in new_vals else " "
            print(f"    {mark} {v:g}  ({len(seen[v])} 处)")
            for w in seen[v][:2 if v in new_vals else 1]:
                print(f"         {w[:110]}")
    if accept:
        import json
        with open(baseline_path, "w", encoding="utf-8") as fh:
            json.dump(fresh, fh, indent=1, sort_keys=True)
        print(f"\n  已写入基线 {baseline_path}；以后只报新出现的组合。")
    elif not issues:
        print("\n  没有新的取值分歧。")
    return issues


def check_crossrefs(ms, supp, figdir):
    """交叉引用是否落空。"""
    print("\n" + "=" * 74)
    print("[2] 交叉引用是否存在")
    print("=" * 74)
    issues = 0

    have_tables = set(int(m) for m in
                      re.findall(r"^## Supplementary Table (\d+)", supp,
                                 flags=re.M))
    have_notes = set(int(m) for m in
                     re.findall(r"^## Supplementary Note (\d+)", supp,
                                flags=re.M))
    have_sfigs = set(int(m) for m in
                     re.findall(r"^## Supplementary Figure (\d+)", supp,
                                flags=re.M))

    cited_t = set(int(m) for m in
                  re.findall(r"Supplementary Tables? (\d+)", ms))
    cited_t |= set(int(m) for m in
                   re.findall(r"Supplementary Tables? \d+[,\s]+(\d+)", ms))
    cited_n = set(int(m) for m in
                  re.findall(r"Supplementary Notes? (\d+)", ms))
    cited_f = set(int(m) for m in
                  re.findall(r"Supplementary Figure (\d+)", ms))

    for label, cited, have in (("Table", cited_t, have_tables),
                               ("Note", cited_n, have_notes),
                               ("Figure", cited_f, have_sfigs)):
        missing = sorted(cited - have)
        unused = sorted(have - cited)
        if missing:
            issues += 1
            print(f"  ★ 正文引用了不存在的 Supplementary {label}: {missing}")
        if unused:
            print(f"    （未被正文引用的 Supplementary {label}: {unused}）")

    # 正文 figure 引用 vs 图注 vs 图片文件
    cap = set(int(m) for m in re.findall(r"\*\*Fig\. (\d+) \|", ms))
    # 引用 = 去掉图注自身那一处之后仍出现的编号
    body = re.sub(r"\*\*Fig\. \d+ \|", "", ms)
    cited_fig = set(int(m) for m in re.findall(r"\bFig\. (\d+)", body))
    if cited_fig - cap:
        issues += 1
        print(f"  ★ 正文引用了没有图注的 Fig: {sorted(cited_fig - cap)}")
    if cap - cited_fig:
        issues += 1
        print(f"  ★ 有图注但正文从未引用的 Fig: {sorted(cap - cited_fig)}")
    files = set()
    if os.path.isdir(figdir):
        for f in os.listdir(figdir):
            m = re.match(r"figure(\d+)_", f)
            if m:
                files.add(int(m.group(1)))
    if cap - files:
        issues += 1
        print(f"  ★ 有图注但没有图片文件的 Fig: {sorted(cap - files)}")
    if files - cap:
        print(f"    （有图片文件但没有图注: {sorted(files - cap)}）")

    # 参考文献编号
    refs = set(int(m) for m in re.findall(r"^(\d+)\.\s", ms, flags=re.M))
    cites = set()
    for m in re.findall(r"\[([\d,\s–-]+)\]", ms):
        for part in re.split(r"[,\s]+", m.strip()):
            if "–" in part or "-" in part:
                a, b = re.split(r"[–-]", part)
                if a.isdigit() and b.isdigit():
                    cites |= set(range(int(a), int(b) + 1))
            elif part.isdigit():
                cites.add(int(part))
    if cites - refs:
        issues += 1
        print(f"  ★ 引用了不存在的参考文献: {sorted(cites - refs)}")
    if refs - cites:
        print(f"    （列出但未引用的参考文献: {sorted(refs - cites)}）")
    if not issues:
        print("  所有交叉引用都能落地。")
    return issues


def check_numbers_against_tables(root, ms, supp):
    """正文与补充材料里出现的三位小数，是否至少在某张结果表里出现过。

    这是粗筛：报出「稿子里写了但任何表都找不到」的数值，多半是打字错误或
    过期数字。噪声会有（换算后的百分比、比值），所以只报值本身。
    """
    print("\n" + "=" * 74)
    print("[3] 稿子里的三位小数能否在结果表中找到")
    print("=" * 74)
    import csv as _csv
    pool = set()
    for dirpath, _, files in os.walk(root):
        if "results_" not in dirpath and not dirpath.endswith("results"):
            continue
        for f in files:
            if not f.endswith(".csv"):
                continue
            try:
                with open(os.path.join(dirpath, f), encoding="utf-8-sig") as fh:
                    for row in _csv.reader(fh):
                        for cell in row:
                            try:
                                v = float(cell)
                            except ValueError:
                                continue
                            pool.add(round(v, 3))
                            pool.add(round(v * 100, 1))
            except Exception:
                continue

    missing = defaultdict(list)
    for docname, text in (("manuscript", ms), ("supplementary", supp)):
        for sect, body in split_sections(text):
            for m in re.finditer(r"(?<![\d.])0\.(\d{3})(?![\d])", body):
                v = round(float(m.group(0)), 3)
                if v in pool:
                    continue
                ctx = " ".join(body[max(0, m.start() - 60):m.end() + 30].split())
                missing[v].append(f"{docname}/{sect}: …{ctx}…")
    if not missing:
        print("  每个三位小数都能在某张表里找到。")
        return 0
    print(f"  {len(missing)} 个数值在任何结果表中都找不到（可能是换算值，"
          f"也可能是错字）：")
    for v, where in sorted(missing.items()):
        print(f"\n    {v}")
        for w in where[:2]:
            print(f"      {w[:120]}")
    return 0          # 粗筛，不计为错误


def check_todo_and_boilerplate(ms, supp):
    print("\n" + "=" * 74)
    print("[4] 投稿前必须清掉的东西")
    print("=" * 74)
    issues = 0
    for name, text in (("manuscript", ms), ("supplementary", supp)):
        for pat, what in ((r"\[\[TODO", "TODO 标记"),
                          (r"Format note", "格式备注"),
                          (r"\(\d+ words\)", "词数标注"),
                          (r"\[\[verify\]\]", "待核实标记"),
                          (r"\bTBD\b|\bXXX\b", "占位符")):
            n = len(re.findall(pat, text))
            if n:
                issues += 1
                print(f"  ★ {name}: {n} 处{what}")
    if not issues:
        print("  干净。")
    return issues


def check_word_limits(ms):
    print("\n" + "=" * 74)
    print("[5] 篇幅")
    print("=" * 74)
    ab = ms.split("## Abstract")[1].split("## Main")[0]
    ab = re.sub(r"\*\(.*?\)\*", "", ab)
    main = re.split(r"^## Main$", ms, flags=re.M)[1].split("## Methods")[0]
    main = re.sub(r"\*\*\[\[TODO.*?\]\]\*\*", "", main, flags=re.S)
    main = re.sub(r"^\|.*$", "", main, flags=re.M)
    main = re.sub(r"\*\*Table 1 \|.*?\*\*", "", main)
    nab = len(re.findall(r"\S+", ab))
    nmain = len(re.findall(r"\S+", main))
    ndisp = len(re.findall(r"\*\*Fig\. \d+ \|", ms)) + \
        len(re.findall(r"\*\*Table \d+ \|", ms))
    bad = 0
    for label, got, lim in (("abstract", nab, 150), ("main text", nmain, 3500),
                            ("display items", ndisp, 6)):
        mark = "★" if got > lim else " "
        if got > lim:
            bad += 1
        print(f"  {mark} {label:<15}{got:>6}  (limit {lim})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--manuscript", default="manuscript.md")
    ap.add_argument("--supplementary", default="supplementary.md")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--section", default=None)
    ap.add_argument("--accept", action="store_true",
                    help="把当前的取值组合写入基线，以后只报新增的")
    ap.add_argument("--baseline", default="docs/cross_audit_baseline.json")
    a = ap.parse_args()

    mp = os.path.join(a.root, a.manuscript)
    sp = os.path.join(a.root, a.supplementary)
    for f in (mp, sp):
        if not os.path.isfile(f):
            sys.exit(f"找不到 {f}")
    ms = open(mp, encoding="utf-8").read()
    supp = open(sp, encoding="utf-8").read()

    import json
    bp = os.path.join(a.root, a.baseline)
    baseline = {}
    if os.path.isfile(bp):
        baseline = {k: [float(x) for x in v]
                    for k, v in json.load(open(bp, encoding="utf-8")).items()}

    total = 0
    if a.section in (None, "numbers"):
        total += check_concepts({"manuscript": ms, "supplementary": supp},
                                baseline, a.accept, bp)
    if a.section in (None, "refs"):
        total += check_crossrefs(ms, supp, os.path.join(a.root, a.figures))
    if a.section in (None, "tables"):
        check_numbers_against_tables(a.root, ms, supp)
    if a.section in (None, "todo"):
        total += check_todo_and_boilerplate(ms, supp)
    if a.section in (None, "length"):
        total += check_word_limits(ms)

    print("\n" + "=" * 74)
    print(f"需要人看一眼的地方: {total}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
