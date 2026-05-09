"""
Stage 2 — DPO trainer for Katherine k0, on top of the SFT adapter.

180 curated chosen/rejected pairs from k0_dpo_only.jsonl (system-stripped).
The "chosen" are Katherine-y responses (direct, embodied, in-character).
The "rejected" are assistant-register responses (helpful, bullet lists, sympathetic).

Hyperparameters:
  epochs       = 2
  lr           = 5e-6     (much smaller than SFT — DPO needs gentle steps)
  beta         = 0.1      (KL strength to reference model; standard)
  batch        = 4 per device, grad_accum = 2 (effective 8)
  max_seq      = 1024
  optim        = adamw_8bit

Loads the SFT adapter as both policy + reference, then trains the policy to
prefer chosen responses over rejected.
"""
import argparse
import os
import sys

from unsloth import FastModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def fmt_dpo_example(ex, tokenizer):
    """Convert {prompt: msgs, chosen: msgs, rejected: msgs} into the
    string-form DPOTrainer expects: prompt str, chosen str, rejected str.
    The chosen/rejected strings are JUST the assistant turn content (no role markers).
    """
    prompt_str = tokenizer.apply_chat_template(
        ex["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    chosen_content = ex["chosen"][0].get("content", "") if ex["chosen"] else ""
    rejected_content = ex["rejected"][0].get("content", "") if ex["rejected"] else ""
    return {
        "prompt": prompt_str,
        "chosen": chosen_content,
        "rejected": rejected_content,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/k0_dpo_curated.jsonl")
    p.add_argument("--sft-adapter", default="adapters/k0_sft_adapter",
                   help="SFT adapter to load as policy starting point.")
    p.add_argument("--output", default="adapters/k0_dpo_adapter")
    p.add_argument("--max_seq", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    if args.skip_train:
        if not os.path.isdir(args.output):
            print(f"[error] --skip-train set but DPO adapter dir not found: {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"[skip-train] DPO adapter at {args.output}")
        return

    if not os.path.isdir(args.sft_adapter):
        print(f"[error] SFT adapter not found at {args.sft_adapter}", file=sys.stderr)
        print(f"        Run finetune_k0.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[load] base + SFT adapter: {args.sft_adapter}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.sft_adapter,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Re-attach LoRA params from SFT adapter as the policy.
    # Reference model is the same SFT-snapshot frozen; DPOTrainer handles that internally.

    print(f"[data] loading {args.data}")
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"[data] {len(ds)} preference pairs loaded")

    ds = ds.map(lambda ex: fmt_dpo_example(ex, tokenizer))

    print("[sample] first DPO example:")
    print("-" * 60)
    print("prompt:  ", ds[0]["prompt"][:500])
    print("chosen:  ", ds[0]["chosen"][:300])
    print("rejected:", ds[0]["rejected"][:300])
    print("-" * 60)

    dpo_config = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        max_length=args.max_seq,
        max_prompt_length=args.max_seq // 2,
        beta=args.beta,
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                     # uses adapter-disabled forward as reference (PEFT trick)
        args=dpo_config,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    print()
    print(f"[train] DPO: {args.epochs} epochs × {len(ds)} pairs / "
          f"effective_batch {args.batch * args.grad_accum}")
    trainer.train()

    print()
    print(f"[save] writing DPO adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[done] DPO stage complete")


if __name__ == "__main__":
    main()
