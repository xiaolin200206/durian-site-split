#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_shadowing.py -- 找出（并可选修复）遮蔽 Python 标准库的文件。

起因: 项目根目录有个自写的 inspect.py。任何 import inspect 都会命中它，
numpy 在导入时调用 inspect.cleandoc 就炸:

    AttributeError: module 'inspect' has no attribute 'cleandoc'

这个坑卡了一周，而且因为那个脚本开头就打印目录清单，任何 import 它的
程序都会先吐一屏无关输出，让人以为是别的问题。

本脚本:
  [1] 扫描项目树，找出与标准库同名的 .py 文件和包目录
  [2] 检查它们是否真的在 import 路径上（同目录的才会遮蔽）
  [3] --fix 重命名为 <name>_local.py 并清掉 __pycache__
  [4] 复验 numpy / pandas 等能否正常导入

用法:
    python fix_shadowing.py                     # 只报告
    python fix_shadowing.py --fix               # 重命名 + 清缓存
    python fix_shadowing.py --root "C:\\path"   # 指定项目根
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
             ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
             "site-packages", "dist-packages"}

# 导入后立刻验证的第三方包
VERIFY = ["numpy", "pandas", "yaml", "PIL"]


def stdlib_names():
    """当前解释器的标准库模块名。优先用 sys.stdlib_module_names。"""
    names = set(getattr(sys, "stdlib_module_names", ()))
    if names:
        return {n for n in names if not n.startswith("_")}
    # 老版本 Python 的退路：扫标准库目录
    d = sysconfig.get_paths().get("stdlib", "")
    out = set()
    if d and os.path.isdir(d):
        for e in os.listdir(d):
            if e.endswith(".py"):
                out.add(e[:-3])
            elif os.path.isdir(os.path.join(d, e)) and \
                    os.path.isfile(os.path.join(d, e, "__init__.py")):
                out.add(e)
    return {n for n in out if not n.startswith("_")}


def scan(root, names):
    """返回 [(路径, 模块名, 类型)]。"""
    hits = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        # 同名的包目录
        for d in list(dn):
            if d in names and os.path.isfile(os.path.join(dp, d,
                                                          "__init__.py")):
                hits.append((os.path.join(dp, d), d, "package"))
        for f in fn:
            if f.endswith(".py") and f[:-3] in names:
                hits.append((os.path.join(dp, f), f[:-3], "module"))
    return hits


def risk(path, root):
    """遮蔽只在该文件与被执行脚本同目录、或该目录在 sys.path 上时发生。
    项目根目录风险最高，子目录只在从那里运行脚本时才有影响。"""
    p = Path(path).parent.resolve()
    r = Path(root).resolve()
    if p == r:
        return "高", "与项目根同级，从根目录运行任何脚本都会命中"
    if str(p) in [str(Path(x).resolve()) for x in sys.path if x]:
        return "高", "该目录在 sys.path 上"
    return "中", "仅当从该目录运行脚本时生效"


def verify_imports():
    """在干净的子进程里验证，避免本进程已缓存的模块影响判断。"""
    code = (
        "import importlib, sys\n"
        "for m in %r:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        v = getattr(mod, '__version__', '?')\n"
        "        print('  ok   %%-10s %%s' %% (m, v))\n"
        "    except Exception as e:\n"
        "        print('  FAIL %%-10s %%s: %%s' %% (m, type(e).__name__, e))\n"
        "import inspect\n"
        "print('  inspect ->', inspect.__file__)\n"
    ) % (VERIFY,)
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print("  stderr:", r.stderr.strip()[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fix", action="store_true",
                    help="重命名为 <name>_local.py 并清 __pycache__")
    ap.add_argument("--suffix", default="_local")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1

    names = stdlib_names()
    print(f"根目录 : {root}")
    print(f"解释器 : {sys.version.split()[0]}  ({sys.executable})")
    print(f"标准库模块名 {len(names)} 个\n")

    print("=" * 72)
    print("[1] 导入现状（修复前）")
    print("=" * 72)
    verify_imports()

    hits = scan(root, names)
    print("\n" + "=" * 72)
    print(f"[2] 与标准库同名的文件 / 目录: {len(hits)} 个")
    print("=" * 72)
    if not hits:
        print("  没有发现。")
    high = []
    for path, mod, kind in sorted(hits):
        lv, why = risk(path, root)
        rel = os.path.relpath(path, root)
        print(f"  [{lv}] {mod:<14} {kind:<8} {rel}")
        print(f"        {why}")
        if lv == "高":
            high.append((path, mod, kind))

    caches = []
    for dp, dn, fn in os.walk(root):
        if os.path.basename(dp) == "__pycache__":
            caches.append(dp)
    if caches:
        print(f"\n  __pycache__ 目录 {len(caches)} 个"
              f"（含遮蔽模块的编译缓存时同样会命中）")

    if not args.fix:
        if high:
            print("\n" + "=" * 72)
            print("加 --fix 执行重命名：")
            for path, mod, _ in high:
                print(f"  {os.path.basename(path)} -> "
                      f"{mod}{args.suffix}.py")
            print("\n  重命名后，原来 import 这些模块的自己的代码要跟着改。")
        return 0

    # ---------------- 修复 ----------------
    print("\n" + "=" * 72)
    print("[3] 执行修复")
    print("=" * 72)
    done = 0
    for path, mod, kind in high:
        p = Path(path)
        new = p.with_name(mod + args.suffix + (".py" if kind == "module"
                                               else ""))
        if new.exists():
            print(f"  跳过 {p.name}: {new.name} 已存在")
            continue
        try:
            p.rename(new)
            print(f"  {p.name} -> {new.name}")
            done += 1
        except OSError as e:
            print(f"  ★ 重命名失败 {p.name}: {e}")

    for c in caches:
        try:
            shutil.rmtree(c)
        except OSError:
            pass
    print(f"  清掉 {len(caches)} 个 __pycache__")
    print(f"  共重命名 {done} 个")

    print("\n" + "=" * 72)
    print("[4] 导入现状（修复后）")
    print("=" * 72)
    verify_imports()
    print("\n  inspect 若已指向解释器自带的路径，问题解决。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
