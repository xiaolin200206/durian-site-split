#!/bin/bash
# 第 2 步：回来、QC 完标注之后。把 PWA 导出合成 6 类测试集（不用 GPU）。只能运行一次！
# 文件名要先改成 pg1t1_…、pg1t2_…（树号在 t 后面），没有树号的图会被拒收。
# PWA 导出的文件夹要有 images/ 和 labels/（YOLO txt）和 data.yaml 或 classes.txt。
# 如果导出的是 Roboflow zip，把 --folder 换成 --zips /path/to/zips 并去掉 --resize640。
SITE=penang
EXPORT=/root/autodl-tmp/durian/pwa_export/$SITE
cd /root/autodl-tmp/durian
python prospective/prospective_site.py prepare --site $SITE --folder "$EXPORT" --resize640
echo "看一眼 prospective/${SITE}_prepare_report.txt：树的分组对不对、每类张数合不合理、去掉的框是不是只有六类之外的。"
