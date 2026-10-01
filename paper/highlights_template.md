# Highlights

- Random splits gave {{headline_range/ratio_min:.1f}}–{{headline_range/ratio_max:.1f}}× the unseen-farm mAP50 of durian detectors
- The held-out farm dominated; mean detector scores differed by at most {{model_pairs/max_abs_mean_diff:.3f}} mAP50
- Psyllid classes kept {{per_class_summary/psyllid_retained_range/0:.0f}}–{{per_class_summary/psyllid_retained_range/1:.0f}}% of their AP on a new farm; other classes kept {{per_class_summary/other_retained_range/0:.0f}}–{{per_class_summary/other_retained_range/1:.0f}}%
- At fixed training length, more farms and more images gave indistinguishable gains
- {{highlight_rq3}}
