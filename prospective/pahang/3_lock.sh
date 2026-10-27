#!/bin/bash
# 第 3 步：锁定测试集（不用 GPU）。只能运行一次！
# 先把下面的数字改成真实数字（QC 记账的汇总）。
N_PHOTOS=0          # 现场拍的总张数
N_ANNOTATED=0       # 标注后进入测试集的张数
N_ADDED=0           # QC 时加的框
N_REMOVED=0         # QC 时删的框
N_RELABELLED=0      # QC 时改类的框
N_SCRATCH=40        # 关掉预测框从零标的张数
SITE=pahang
cd /root/autodl-tmp/durian
python prospective/prospective_site.py lock --site $SITE --note "Photographs taken: ${N_PHOTOS}; annotated and retained: ${N_ANNOTATED}. Pre-labels from the PWA detector v0 were corrected by the author: ${N_ADDED} boxes added, ${N_REMOVED} removed, ${N_RELABELLED} relabelled. ${N_SCRATCH} images were additionally annotated from scratch with model output hidden (agreement reported separately)."
echo "把上面的锁定文件 SHA-256 发邮件给自己（或 commit），再运行 4_score.sh"
