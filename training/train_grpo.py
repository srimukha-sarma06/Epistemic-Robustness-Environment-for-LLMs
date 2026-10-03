"""
train_grpo.py — GRPO fine-tuning on the Epistemic Robustness Environment.

The environment's graders are the reward functions (called in-process).
Defaults target a single ~6 GB consumer GPU: a 0.5B instruct model with LoRA.

    pip install -e ".[train]"                       # trl, peft, datasets, torch
    python -m training.train_grpo --dry-run         # build data, print stats, no training
    python -m training.train_grpo                   # train (LoRA, Qwen2.5-0.5B-Instruct)
    python -m training.train_grpo --model Qwen/Qwen2.5-1.5B-Instruct --num-generations 8

The held-out split (by sample, never seen in training) is written to
<output-dir>/split.json; evaluate before and after with:

    python -m training.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --split-file runs/grpo/split.json
    python -m training.evaluate --model runs/grpo --split-file runs/grpo/split.json

To run the trained model through the original pipeline (inference.py, full
multi-turn episodes), serve it behind any OpenAI-compatible server (e.g.
`vllm serve runs/grpo-merged`) and point API_BASE_URL / MODEL_NAME at it.
"""

import argparse
import json
import os
import sys
from collections import Counter

from server.models import TaskName
from training.grpo_data import build_examples, epistemic_reward, split_by_sample


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output-dir", default="runs/grpo")
    p.add_argument("--tasks", nargs="+", choices=[t.value for t in TaskName], default=None,
                   help="train on these tasks only (default: all five)")

    data = p.add_argument_group("data")
    data.add_argument("--seeds-per-claim", type=int, default=4, help="pushback variations per claim")
    data.add_argument("--max-turn", type=int, default=3, help="deepest pressure turn to train (0 = all)")
    data.add_argument("--single-turn-repeats", type=int, default=4)
    data.add_argument("--eval-fraction", type=float, default=0.2)

    grpo = p.add_argument_group("GRPO")
    grpo.add_argument("--num-generations", type=int, default=4, help="completions per prompt (group size)")
    grpo.add_argument("--batch-size", type=int, default=4, help="per-device train batch size")
    grpo.add_argument("--grad-accum", type=int, default=4)
    grpo.add_argument("--epochs", type=float, default=1.0)
    grpo.add_argument("--lr", type=float, default=None, help="default: 1e-5 with LoRA, 1e-6 without")
    grpo.add_argument("--beta", type=float, default=0.04,
                      help="KL penalty to the reference model (TRL's default is 0; keep >0 to limit reward hacking)")
    grpo.add_argument("--temperature", type=float, default=1.0)
    grpo.add_argument("--max-completion-length", type=int, default=256)
    grpo.add_argument("--seed", type=int, default=0)

    model = p.add_argument_group("model / hardware")
    model.add_argument("--no-lora", action="store_true", help="full fine-tuning instead of LoRA")
    model.add_argument("--lora-r", type=int, default=16)
    model.add_argument("--lora-alpha", type=int, default=32)
    model.add_argument("--gradient-checkpointing", action="store_true")
    model.add_argument("--use-vllm", action="store_true", help="generate with vLLM (colocated)")
    model.add_argument("--report-to", default="none", help="e.g. wandb, tensorboard")
    model.add_argument("--logging-steps", type=int, default=5)
    model.add_argument("--save-steps", type=int, default=200)

    p.add_argument("--dry-run", action="store_true", help="build and summarise data only")
    return p.parse_args(argv)


def build_split(args):
    examples = build_examples(
        seeds_per_claim=args.seeds_per_claim,
        max_turn=args.max_turn or None,
        single_turn_repeats=args.single_turn_repeats,
        tasks=args.tasks,
    )
    return split_by_sample(examples, eval_fraction=args.eval_fraction)


def describe(train, evals) -> str:
    lines = [f"train examples: {len(train)}   eval examples: {len(evals)}"]
    by_task = Counter(r["task"] for r in train)
    by_turn = Counter((r["task"], r["turn"]) for r in train if r["kind"] == "claim")
    for task, n in sorted(by_task.items()):
        turns = ", ".join(f"t{t}={c}" for (tk, t), c in sorted(by_turn.items()) if tk == task)
        lines.append(f"  {task:<24} {n:>5}  {turns}")
    lines.append(f"held-out samples: {len({r['sample_id'] for r in evals})}")
    return "\n".join(lines)


def write_split(output_dir: str, train, evals, args) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "split.json")
    with open(path, "w") as f:
        json.dump({
            "train_sample_ids": sorted({r["sample_id"] for r in train}),
            "eval_sample_ids":  sorted({r["sample_id"] for r in evals}),
            "data_args": {k: getattr(args, k) for k in
                          ("seeds_per_claim", "max_turn", "single_turn_repeats", "eval_fraction", "tasks")},
        }, f, indent=2)
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    train, evals = build_split(args)
    print(describe(train, evals), flush=True)

    if args.dry_run:
        sample = train[0]
        print("\nexample prompt:")
        for m in sample["prompt"]:
            print(f"  [{m['role']}] {m['content'][:160]}")
        return 0

    gen_batch = args.batch_size * args.grad_accum
    if gen_batch % args.num_generations:
        sys.exit(f"batch-size × grad-accum ({gen_batch}) must be divisible by num-generations "
                 f"({args.num_generations})")

    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        sys.exit(f"Missing training dependency ({e.name}). Install with: pip install -e \".[train]\"")

    use_lora = not args.no_lora
    lr = args.lr if args.lr is not None else (1e-5 if use_lora else 1e-6)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.use_vllm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        log_completions=True,
        num_completions_to_print=2,
        seed=args.seed,
    )

    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=epistemic_reward,
        args=config,
        train_dataset=Dataset.from_list(train),
        peft_config=peft_config,
    )
    split_path = write_split(args.output_dir, train, evals, args)
    print(f"split written to {split_path}", flush=True)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
