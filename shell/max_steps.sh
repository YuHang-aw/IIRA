for S in 1 2 3 4; do
  python runners/eval_baselines.py \
    --dataset artifacts/rl/rl_vindr.jsonl \
    --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
    --mode policy_init --n_eval 500 --max_steps $S \
    --out_dir artifacts/eval/curve_pi/steps_$S
done
