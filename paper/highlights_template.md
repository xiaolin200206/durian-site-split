<!-- DRAFT: re-check highlights 4-5 against the v2 results; delete this line when done -->
# Highlights

- Detectors scored {{headline_range/ratio_min:.1f}}–{{headline_range/ratio_max:.1f}} times lower mAP50 on durian farms never seen in training
- The held-out farm set the score: six detectors differed by ≤ {{model_pairs/max_abs_mean_diff:.3f}} mAP50 on average
- Psyllid classes kept {{per_class_summary/psyllid_retained_range/0:.0f}}–{{per_class_summary/psyllid_retained_range/1:.0f}}% of their AP on a new farm; other classes kept {{per_class_summary/other_retained_range/0:.0f}}–{{per_class_summary/other_retained_range/1:.0f}}%
- With training length fixed, more farms and more images per farm both helped
- {{highlight_rq3}}
