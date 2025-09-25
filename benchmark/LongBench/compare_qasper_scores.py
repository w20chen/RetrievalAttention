import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from metrics import qa_f1_score

# 路径配置
retroinfer_path = "results/pred/llama-3-8b-1048k/RetroInfer/qasper.jsonl"
vllm_path = "results/pred/llama-3-8b-1048k/vllm/qasper.jsonl"

# 读取数据
with open(retroinfer_path, "r", encoding="utf-8") as f:
    retroinfer_data = [json.loads(line) for line in f]
with open(vllm_path, "r", encoding="utf-8") as f:
    vllm_data = [json.loads(line) for line in f]

assert len(retroinfer_data) == len(vllm_data), "Difference in numbers of samples!"

print(f"{len(retroinfer_data)} samples in total.\n")

for idx, (r, v) in enumerate(zip(retroinfer_data, vllm_data)):
    gt_list = r["answers"]
    r_pred = r["pred"]
    v_pred = v["pred"]
    # 取最大f1分数
    r_score = max(qa_f1_score(r_pred, gt) for gt in gt_list)
    v_score = max(qa_f1_score(v_pred, gt) for gt in gt_list)
    print(f"Sample #{idx+1}: RetroInfer={r_score:.4f}, vLLM={v_score:.4f}")
    print(f"  GT: {gt_list}")
    print(f"  RetroInfer: {r_pred}")
    print(f"  vLLM: {v_pred}\n")
