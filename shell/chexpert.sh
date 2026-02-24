# python runners/prep_chexpert_to_jsonl.py \
#   --root /home/neutron/sdc/RAG/data/chexpertchestxrays-u20210408 \
#   --csv train_visualCheXbert.csv \
#   --out artifacts/transfer/chexpert_vindr6_50p50n.jsonl \
#   --frontal_only \
#   --uncertain_policy ignore \
#   --limit_pos 500 --limit_neg 500


  python -m runners.main eval3 \
  --data artifacts/transfer/chexpert_vindr6_50p50n.jsonl \
  --model /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --lora artifacts/ckpt_rl/best \
  --max_steps 3 --n_eval 999999

python runners/eval_baselines.py \
  --dataset artifacts/transfer/chexpert_vindr6_50p50n.jsonl \
  --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --mode policy_init --n_eval 999999 --max_steps 3
