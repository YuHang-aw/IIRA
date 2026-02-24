
# 1) 训练视觉头（冻结 BiomedCLIP）
python -m runners.main train_head \
  --vindr_root /home/neutron/sdc/RAG/data/vinDr-CXR \
  --out artifacts/vhead/biomedclip \
  --batch_size 24 \
  --epochs 2 \
  --biomedclip_dir /home/neutron/sdc/MODEL/BiomedCLIP-PubMedBERT

# # 2) 跑基线三件套（你前面加过的 eval3）
# python -m runners.main eval3 --data /path/to/VinDr-CXR --model Qwen/Qwen2.5-VL-3B --max_steps 3
# # 输出 release.json 会含：brier_no_tool / brier_tool / evidence_lift / policy_init{...}
