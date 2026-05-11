"""
Stage 2 (v2) — DPO trainer for Katherine K0 v2 on Qwen3.5-9B,
WITH vision preserved, on top of the v2 SFT adapter.

Differences from dpo_k0.py (v1):

  1. Uses unsloth.FastVisionModel instead of unsloth.FastModel.
     v1 used FastModel which silently dropped the vision tower from
     the merged GGUF. FastVisionModel keeps it.
  2. Sets finetune_vision_layers=False to preserve native vision tower.
  3. Default data path points at v2 cumulative DPO
     (dataset/v2/cumulative/tier_*/dpo_train.jsonl).
  4. Default reference adapter points at v2 SFT output
     (adapters/k0_v2_sft_t*).
  5. Handles v2 DPO schema:
        {messages: [...], chosen: str, rejected: str, _cat: str, _type: "dpo"}
     vs v1 schema: {prompt: [msgs], chosen: [msg], rejected: [msg]}

Same hyperparameters as v1 otherwise: epochs=2, lr=5e-6, beta=0.1,
batch=4, grad_accum=2, max_seq=1024, bf16, adamw_8bit.

Wallclock target: ~20-30 min on H200, ~$1-2 (DPO is faster than SFT — fewer
steps, smaller dataset).

Usage:
    python scripts/dpo_k0_v2.py --tier 500
    python scripts/dpo_k0_v2.py --tier 1000
    python scripts/dpo_k0_v2.py --data dataset/v2/cumulative/tier_500/dpo_train.jsonl \\
        --sft-adapter adapters/k0_v2_sft_t500 --output adapters/k0_v2_dpo_t500
"""
import argparse
import os
import sys

from unsloth import FastVisionModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def fmt_dpo_example_v2(ex, tokenizer):
    """Convert v2 DPO schema {messages: list, chosen: str, rejected: str}
    into the string-form DPOTrainer expects: prompt str, chosen str, rejected str.

    The v2 messages array already ends with the user turn (per
    trace_generation_prompt.md DPO format). chosen and rejected are already
    plain strings (the candidate K0 final replies)."""
    prompt_str = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "prompt": prompt_str,
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
    }


def resolve_paths(tier, data, sft_adapter, output):
    """If --tier is given, derive default paths. Explicit args override."""
    if tier is not None:
        if data is None:
            data = f"dataset/v2/cumulative/tier_{tier}/dpo_train.jsonl"
        if sft_adapter is None:
            sft_adapter = f"adapters/k0_v2_sft_t{tier}"
        if output is None:
            output = f"adapters/k0_v2_dpo_t{tier}"
    return data, sft_adapter, output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=None,
                   help="Tier shorthand: derives data/sft-adapter/output paths.")
    p.add_argument("--data", default=None,
                   help="DPO data JSONL path. Defaults to v2 tier_X/dpo_train.jsonl.")
    p.add_argument("--sft-adapter", default=None,
                   help="v2 SFT adapter to load as policy starting point.")
    p.add_argument("--output", default=None,
                   help="Output dir for DPO adapter.")
    p.add_argument("--max_seq", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--vanilla", action="store_true",
                   help="Strip DPO-VISION-REGISTER and DPO-VOICE-REGISTER at load time. "
                        "Equivalent to --exclude-dpo-types VISION-REGISTER,VOICE-REGISTER. "
                        "Pairs with finetune_k0_v2.py --vanilla for text-only training.")
    p.add_argument("--exclude-dpo-types", default="",
                   help="Comma-separated DPO subtypes to skip at load time, "
                        "e.g. 'VISION-REGISTER,VOICE-REGISTER'. Matches _cat exactly (DPO- prefix added if missing).")
    args = p.parse_args()

    data, sft_adapter, output = resolve_paths(args.tier, args.data, args.sft_adapter, args.output)
    if not data or not sft_adapter or not output:
        print("[error] either --tier or all of --data/--sft-adapter/--output required",
              file=sys.stderr)
        sys.exit(1)

    if args.skip_train:
        if not os.path.isdir(output):
            print(f"[error] --skip-train set but DPO adapter dir not found: {output}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[skip-train] DPO adapter at {output}")
        return

    if not os.path.isdir(sft_adapter):
        print(f"[error] v2 SFT adapter not found at {sft_adapter}", file=sys.stderr)
        print(f"        Run scripts/finetune_k0_v2.py first.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(data):
        print(f"[error] DPO data not found at {data}", file=sys.stderr)
        print(f"        Run scripts/build_cumulative.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[load] base + v2 SFT adapter: {sft_adapter}  (FastVisionModel — preserves vision tower)")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=sft_adapter,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Re-attach LoRA params from v2 SFT adapter as the policy.
    # finetune_vision_layers=False matches the v2 SFT setting (we don't want to
    # disturb the native vision tower; only train the language pathway).
    # Reference model uses adapter-disabled forward (PEFT trick); ref_model=None
    # in DPOTrainer.

    print(f"[data] loading {data}")
    ds = load_dataset("json", data_files=data, split="train")
    print(f"[data] {len(ds)} preference pairs loaded (v2 schema)")

    # Optional: strip DPO subtypes (e.g., DPO-VISION-REGISTER, DPO-VOICE-REGISTER)
    # at load time. --vanilla is shorthand for --exclude-dpo-types VISION-REGISTER,
    # VOICE-REGISTER (text-only DPO, no vision or voice register pairs).
    excluded_dpo = set()
    if args.vanilla:
        excluded_dpo.update(['DPO-VISION-REGISTER', 'DPO-VOICE-REGISTER'])
    if args.exclude_dpo_types:
        excluded_dpo.update(
            ('DPO-' + t.strip().upper().lstrip('DPO-').lstrip('-'))
            for t in args.exclude_dpo_types.split(',')
        )
    if excluded_dpo:
        before = len(ds)
        ds = ds.filter(lambda ex: ex.get('_cat', '') not in excluded_dpo)
        after = len(ds)
        print(f"[filter] excluded DPO subtypes {sorted(excluded_dpo)}: {before} -> {after} pairs (-{before-after})")

    ds = ds.map(lambda ex: fmt_dpo_example_v2(ex, tokenizer))

    print("[sample] first DPO example:")
    print("-" * 60)
    print("prompt:  ", ds[0]["prompt"][:500])
    print("chosen:  ", ds[0]["chosen"][:300])
    print("rejected:", ds[0]["rejected"][:300])
    print("-" * 60)

    dpo_config = DPOConfig(
        output_dir=output,
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
    print(f"[train] DPO v2: {args.epochs} epochs × {len(ds)} pairs / "
          f"effective_batch {args.batch * args.grad_accum}")
    trainer.train()

    print()
    print(f"[save] writing v2 DPO adapter to {output}")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[done] v2 DPO stage complete")


if __name__ == "__main__":
    main()
