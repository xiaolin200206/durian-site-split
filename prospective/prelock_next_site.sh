#!/bin/bash
# Pahang 的预锁定已经在 2026-10-26 做好了：prospective/pahang_prelock.json（不要再跑这个脚本）。
# 它沿用 Seremban 锁定文件里那 200 个 LOFO 模型的哈希，预标模型是 v5（SHA-256 420b4c52…1dce）。
# 下一个站（Penang）的预锁定请复制这个文件，把 SITE 改成 penang，PWA_WEIGHTS 换成那时线上的模型，
# 在 GPU 机器上跑（不加 --from-lock），或者像 Pahang 一样加 --from-lock prospective/prospective_lock.json。
SITE=penang
PWA_WEIGHTS=/root/autodl-tmp/durian/pwa/REPLACE_WITH_MODEL_IN_USE.onnx
PWA_VERSION="SnapDurian server, model REPLACE"
cd /root/autodl-tmp/durian
python prospective/prospective_site.py prelock --site $SITE \
  --pwa-weights "$PWA_WEIGHTS" --pwa-version "$PWA_VERSION" \
  --tree-rule odd \
  --from-lock prospective/prospective_lock.json \
  --note-file prospective/$SITE/prelock_note.txt
echo "把上面的预锁定文件 SHA-256 发邮件给自己，或 git add prospective/${SITE}_prelock.json && git commit && git push。然后才出发。"
