#!/bin/bash
# 第 1 步：出发前预锁定（不用 GPU）。只能运行一次！
# 先改下面三行：PWA 里 v0 的权重文件路径、PWA 版本号；prelock_note.txt 里带 [EDIT] 的地方也先改好。
PWA_WEIGHTS=/root/autodl-tmp/durian/pwa/v0.onnx
PWA_VERSION="snapdurian-pwa v0 (build 2026-10-xx)"
SITE=pahang
cd /root/autodl-tmp/durian
python prospective/prospective_site.py prelock --site $SITE \
  --pwa-weights "$PWA_WEIGHTS" --pwa-version "$PWA_VERSION" \
  --note-file prospective/$SITE/prelock_note.txt
echo "把上面的预锁定文件 SHA-256 发邮件给自己，或 git add prospective/${SITE}_prelock.json && git commit && git push。然后才出发。"
