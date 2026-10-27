#!/bin/bash
# 第 4 步：打分（要 GPU）。跑完自动关机，可以挂着去睡觉：
#   nohup bash prospective/penang/4_score.sh > prospective/penang_score.log 2>&1 &
# 不想自动关机就删掉最后一行。
SITE=penang
cd /root/autodl-tmp/durian
python prospective/prospective_site.py score --site $SITE --device 0
shutdown
