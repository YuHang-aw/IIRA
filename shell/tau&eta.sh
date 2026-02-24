# # gate 扫 tau
# for T in 0.02 0.06 0.12; do
#   python runners/eval_baselines.py \
#     --dataset artifacts/rl/vindr_200.jsonl \
#     --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
#     --mode policy_init --tool_mode kbcs --calibrate_tool \
#     --gate gate --tau $T --eta 0.5 --n_eval 200 --max_steps 3 \
#     --out_dir artifacts/sweeps/gate_tau_${T}_init
# done

# # mix 扫 eta
# for E in 0.1 0.25 0.5; do
#   python runners/eval_baselines.py \
#     --dataset artifacts/rl/vindr_200.jsonl \
#     --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
#     --mode policy_init --tool_mode kbcs --calibrate_tool \
#     --gate mix --eta $E --n_eval 200 --max_steps 3 \
#     --out_dir artifacts/sweeps/mix_eta_${E}_init
# done
# gate：扫触发阈值 × 采纳阈值
for A in 0.60 0.75 0.90; do
  for T in 0.02 0.06 0.12; do
    python runners/eval_baselines.py \
      --dataset artifacts/rl/vindr_200.jsonl \
      --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
      --mode policy_init --tool_mode kbcs --calibrate_tool \
      --gate gate --tau ${T} --eta 0.5 \
      --check_alpha ${A} --max_checks_per_case 1 \
      --n_eval 200 --max_steps 3 \
      --out_dir artifacts/sweeps/gate_checkA_${A}_tau_${T}_init
  done
done

# mix：扫触发阈值 × 融合权
for A in 0.60 0.75 0.90; do
  for E in 0.10 0.25 0.50; do
    python runners/eval_baselines.py \
      --dataset artifacts/rl/vindr_200.jsonl \
      --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
      --mode policy_init --tool_mode kbcs --calibrate_tool \
      --gate mix --eta ${E} \
      --check_alpha ${A} --max_checks_per_case 1 \
      --n_eval 200 --max_steps 3 \
      --out_dir artifacts/sweeps/mix_checkA_${A}_eta_${E}_init
  done
done
