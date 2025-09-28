import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# -os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

model2path = json.load(open("config/model2path.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(model2path.keys()), help="model name")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--task', type=str, help="task name. work when --e is false")
    parser.add_argument('--num_examples', type=int, default=-1, help="num of example to evaluate. -1 for all.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="vllm gpu memory utilization")
    parser.add_argument('--max_model_len', type=int, default=None, help="vllm max model len (default: from config)")
    return parser.parse_args(args)

def get_pred_vllm(llm, data, max_new_tokens, prompt_format, out_path):
    sampling_params = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=max_new_tokens, 
        stop=["Human:", "Assistant:", "assistant", "User:", "System:", "</s>", "Question:", "Q:", "\n\n", "\\n\\n"]
    )

    tokenizer = llm.get_tokenizer() if hasattr(llm, 'get_tokenizer') else None
    prompt_token_lens = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        if tokenizer is not None:
            token_ids = tokenizer.encode(prompt)
            token_len = len(token_ids)
        else:
            print("Tokenizer not available, skipping token length calculation.")
        prompt_token_lens.append(token_len)
        print(f"Prompt length (chars): {len(prompt)}")
        print(f"Prompt token length: {token_len}")

        outputs = llm.generate([prompt], sampling_params)
        pred = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj.get("answers", []),
                "all_classes": json_obj.get("all_classes", []),
                "length": json_obj.get("length", 0)
            }, f, ensure_ascii=False)
            f.write('\n')
    if prompt_token_lens:
        print(f"Prompt token stats: min={min(prompt_token_lens)}, max={max(prompt_token_lens)}, avg={sum(prompt_token_lens)/len(prompt_token_lens):.2f}")

def main():
    args = parse_args()
    model_name = args.model
    model_path = model2path[model_name]
    max_length = args.max_model_len if args.max_model_len else model2maxlen[model_name]
    num_examples = args.num_examples

    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        if not args.task:
            # raise ValueError("--task must be specified when not using --e")
            datasets = [
                "qasper", "gov_report",
                "triviaqa", "lcc", "repobench-p"
            ]
        else:
            datasets = [args.task]

    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            prefix = f"results/pred_e/{model_name}/vllm"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            prefix = f"results/pred/{model_name}/vllm"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        out_path = f"{prefix}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_all = data_all[:num_examples] if num_examples > 0 else data_all

        llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=min(20000,max_length))
        get_pred_vllm(llm, data_all, max_new_tokens, prompt_format, out_path)

if __name__ == "__main__":
    main()