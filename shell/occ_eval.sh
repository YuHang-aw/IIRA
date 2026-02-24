python eval/eval_occlusion.py \
  --dataset artifacts/rl/rl_vindr.jsonl \
  --out_dir artifacts/eval/occlusion \
  --roi_mode gt --roi_field box --roi_format xyxy \
  --k_rand 25 --mask_value mean --plot


# [occlusion] metrics: {"n_items": 348, "n_skipped": 210, "n_ok": 138, "k_rand": 25, "mask_value": "mean", "roi_mode": "gt", "roi_field": "box", "roi_format": "xyxy", "drop_real_mean": 0.13348174444892616, "drop_rand_mean": 0.09076758622526977, "mean_diff": 0.04271415822365639, "cohens_d": 0.03112085136102301, "mw_auc": 0.5011583700903172}
# [save] artifacts/eval/occlusion

# 指向 release 的 eval_log.jsonl（里面有 {image, concept, roi}）
python eval/eval_occlusion.py \
  --dataset artifacts/releases/20251026-233327/eval_log.jsonl \
  --out_dir artifacts/eval/occlusion_policyROI \
  --roi_mode pred \
  --k_rand 25 --mask_value mean --plot
