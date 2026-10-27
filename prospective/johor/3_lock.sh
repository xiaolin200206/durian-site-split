#!/bin/bash
# 第 3 步：锁定测试集（不用 GPU）。只能运行一次！
# 这一站的预锁定在出发前就做好了，但那时预标模型还没训出来，所以它的哈希在这一步记录。
# 先填两样东西：
#  1) 这一站在 PWA 里实际预标用的模型（服务器上 admin.py models 里标“正在使用”的那个，拷到 GPU 机器上）。
#     导出的 photos.csv 里每张照片的 model 一栏应该都是这个版本；不是的话先停下来。
#  2) QC 记账的汇总数字。
PWA_FILE=/root/autodl-tmp/durian/pwa/durian_11cls_640_v2_XXXXXXXX.onnx   # ← 改成真实文件名
PWA_VERSION=v2                                                        # ← 实际用的版本
N_PHOTOS=0          # 现场拍的总张数
N_ANNOTATED=0       # 标注后进入测试集的张数
N_ADDED=0           # QC 时加的框
N_REMOVED=0         # QC 时删的框
N_RELABELLED=0      # QC 时改类的框
N_SCRATCH=40        # 关掉预测框从零标的张数
VISIT_DATE=YYYY-MM-DD
DISTRICT=""         # 农场所在的县/区
SITE=johor
cd /root/autodl-tmp/durian
python prospective/prospective_site.py lock --site $SITE --pwa-weights "$PWA_FILE" --pwa-version "$PWA_VERSION" --note "Visited ${VISIT_DATE} (${DISTRICT}). Photographs taken: ${N_PHOTOS}; annotated and retained: ${N_ANNOTATED}. Pre-labels from the PWA detector ${PWA_VERSION} were corrected by the author: ${N_ADDED} boxes added, ${N_REMOVED} removed, ${N_RELABELLED} relabelled. ${N_SCRATCH} images were additionally annotated from scratch with model output hidden (agreement reported separately)."
echo "把上面的锁定文件 SHA-256 发邮件给自己（或 commit），再运行 4_score.sh"
