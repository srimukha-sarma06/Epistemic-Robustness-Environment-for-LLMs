"""
evaluate.py — Score a local model on the GRPO prompts with the environment's graders.

Generates one greedy completion per prompt with transformers and grades it with
the same reward function used in training, so base and GRPO-trained models are
compared on identical, held-out prompts.

    python -m training.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --split-file runs/grpo/split.json
    python -m training.evaluate --model runs/grpo --split-file runs/grpo/split.json --output after.json

--model can be a hub name, a full checkpoint directory, or a LoRA adapter
directory (detected via adapter_config.json).

This is a turn-level evaluation (each turn is scored given a fixed history).
For full multi-turn episodes, serve the model behind an OpenAI-compatible
server and use inference.py.
"""

import argparse
import json
import os
import sys

from training.grpo_data import build_examples, completion_text, score_completion, split_by_sample, summarise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--split-file", default=None,
                   help="split.json from train_grpo.py (default: rebuild with default data args)")
    p.add_argument("--which", choices=["eval", "train", "all"], default="eval")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="score at most N prompts (for quick checks)")
    p.add_argument("--output", default=None, help="write per-prompt results + summary JSON here")
    return p.parse_args(argv)


def select_rows(args) -> list[dict]:
    data_args = {"seeds_per_claim": 4, "max_turn": 3, "single_turn_repeats": 4,
                 "eval_fraction": 0.2, "tasks": None}
    eval_ids = None
    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        data_args.update(split["data_args"])
        eval_ids = set(split["eval_sample_ids"])

    rows = build_examples(seeds_per_claim=data_args["seeds_per_claim"],
                          max_turn=data_args["max_turn"] or None,
                          single_turn_repeats=1,          # no point scoring duplicates
                          tasks=data_args["tasks"])
    if eval_ids is None:
        _, held = split_by_sample(rows, eval_fraction=data_args["eval_fraction"])
        eval_ids = {r["sample_id"] for r in held}

    if args.which == "eval":
        rows = [r for r in rows if r["sample_id"] in eval_ids]
    elif args.which == "train":
        rows = [r for r in rows if r["sample_id"] not in eval_ids]
    return rows[: args.limit] if args.limit else rows


def load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(os.path.join(name, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        with open(os.path.join(name, "adapter_config.json")) as f:
            base = json.load(f)["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(name if os.path.exists(os.path.join(name, "tokenizer_config.json")) else base)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer


def generate(model, tokenizer, prompts: list[list[dict]], max_new_tokens: int, batch_size: int) -> list[str]:
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", file=sys.stderr, flush=True)
    return outputs


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = select_rows(args)
    print(f"scoring {len(rows)} prompts ({args.which} split) with {args.model}", file=sys.stderr)

    try:
        model, tokenizer = load_model(args.model)
    except ImportError as e:
        sys.exit(f"Missing dependency ({e.name}). Install with: pip install -e \".[train]\"")

    completions = generate(model, tokenizer, [r["prompt"] for r in rows], args.max_new_tokens, args.batch_size)
    scores = [
        score_completion(completion_text(c), r["kind"], r["sample_id"], r["turn"], r["max_turns"],
                         r["offers_alternative"], r["prev_response"])
        for c, r in zip(completions, rows)
    ]
    summary = summarise(rows, scores)
    overall = round(sum(scores) / len(scores), 3) if scores else 0.0

    print(f"\n{'task':<26}{'n':>5}{'reward':>9}{'pass':>8}")
    for task, s in summary.items():
        print(f"{task:<26}{s['n']:>5}{s['mean_reward']:>9.3f}{s['pass_rate']:>8.0%}")
    print(f"{'overall':<26}{len(scores):>5}{overall:>9.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model, "which": args.which, "overall_mean_reward": overall,
                "summary": summary,
                "results": [{"task": r["task"], "sample_id": r["sample_id"], "turn": r["turn"],
                             "score": s, "completion": c} for r, s, c in zip(rows, scores, completions)],
            }, f, indent=2)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
