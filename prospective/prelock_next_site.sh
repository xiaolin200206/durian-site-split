#!/bin/bash
# Pahang（2026-10-26）、Penang / Malacca / Johor（2026-10-27）的预锁定都已经做好了，不要再跑这个脚本。
# 只有临时加一个站（比如 Perak）才用它：先写好 prospective/$SITE/prelock_note.txt
# （照 johor/prelock_note.txt 改站名、文件名前缀和农场来源），然后在出发前运行。
# 预标模型的哈希不在这里记，在 3_lock.sh 里用 --pwa-weights 记。
SITE=perak
cd /root/autodl-tmp/durian
python prospective/prospective_site.py prelock --site $SITE \
  --tree-rule odd \
  --from-lock prospective/prospective_lock.json \
  --note-file prospective/$SITE/prelock_note.txt
echo "把上面的预锁定文件 SHA-256 发邮件给自己，或 git add prospective/${SITE}_prelock.json && git commit && git push。然后才出发。"
