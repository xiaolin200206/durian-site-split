#!/bin/bash
# 第 3 步：打分（要 GPU，正常开机）。跑完自动关机，可以挂着去睡觉：
#   nohup bash 3_score.sh > score.log 2>&1 &
# 不想自动关机就删掉最后一行。
cd /root/autodl-tmp/durian
python prospective_seremban.py score --device 0
