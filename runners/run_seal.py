# runners/run_seal.py
import sys, json, yaml, pathlib, time, shutil, subprocess
from eval.evaluate_dual_env import evaluate

def main():
    if len(sys.argv) < 2:
        print("Usage: python runners/run_seal.py <data_root|dataset.jsonl>")
        sys.exit(1)
    data_root = sys.argv[1]

    cfg = yaml.safe_load(open("configs/pipeline.yaml", "r"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = pathlib.Path(cfg["PIPELINE"]["RELEASE_DIR"]) / stamp
    rel.mkdir(parents=True, exist_ok=True)

    # 备份关键配置
    shutil.copy("configs/pipeline.yaml", rel / "pipeline.yaml")
    for f in ("configs/calib_kbcs.json", "configs/tool_runtime.json"):
        p = pathlib.Path(f)
        if p.exists(): shutil.copy(p, rel / p.name)

    try:
        gitv = subprocess.getoutput("git rev-parse --short HEAD").strip()
    except Exception:
        gitv = "nogit"
    (rel / "git.txt").write_text(gitv + "\n", encoding="utf-8")

    # 关键：把 log_path 传进 evaluator
    metrics = evaluate(dataset_or_path=data_root, log_path=str(rel / "eval_log.jsonl"))

    (rel / "release.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    print(f"[OK] wrote {rel/'release.json'}")

if __name__ == "__main__":
    main()
