#!/bin/bash
# 第 2 步：锁定（不用 GPU，无卡模式即可）。只能运行一次！
# 先把下面的 N_BEFORE（counts.csv 里 o1t1–o1t7 的总数）和 N_AFTER（prepare 打印的“合并后 X 张”）改成真实数字
N_BEFORE=105
N_AFTER=82
cd /root/autodl-tmp/durian
python prospective_seremban.py lock --note "Before annotation, images whose disease symptoms were too ambiguous to label confidently were removed by the annotator (visual judgement only; no model predictions were viewed). Original photos: ${N_BEFORE}; retained: ${N_AFTER}. This removal favours higher mAP50, i.e. it works against prediction P1."
echo "把上面的锁定文件 SHA-256 发邮件给自己，再运行 3_score.sh"
