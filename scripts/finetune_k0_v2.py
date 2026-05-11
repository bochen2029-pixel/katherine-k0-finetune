"""
Stage 1 (v2) - QLoRA SFT trainer for Katherine K0 v2 on Qwen3.5-9B,
WITH vision preserved.

Differences from finetune_k0.py (v1):

  1. Uses unsloth.FastVisionModel instead of unsloth.FastModel.
     v1 used FastModel which silently dropped the vision tower from
     the merged GGUF. FastVisionModel keeps it.
  2. Sets finetune_vision_layers=False to preserve native vision tower.
     We don't want to disturb what's already working (OCR, image
     understanding). We only fine-tune the language pathway.
  3. Default data path points at the v2 dataset
     (dataset/v2/cumulative/tier_*/sft_train.jsonl).
  4. enable_thinking still False (K0 reasons in prose, not <think>).

Same hyperparameters as v1 otherwise: rank 64, alpha 128, lr 1e-4,
3 epochs, batch 16 grad_accum 2, bf16, adamw_8bit.

Wallclock target: ~50-60 min on H200, ~$3-5. Same as v1.

Usage:
    python scripts/finetune_k0_v2.py --tier 500
    python scripts/finetune_k0_v2.py --tier 1000
    python scripts/finetune_k0_v2.py --data dataset/v2/cumulative/tier_500/sft_train.jsonl --output adapters/k0_v2_sft_t500
"""
import argparse
import os
import sys

# Unsloth MUST import before transformers/peft/trl
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


def do_train(args, model, tokenizer):
    """Attach LoRA (language layers only), format dataset, train, save adapter."""
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,   # preserve native Qwen3.5-9B vision tower
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"[data] loading {args.data}")
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"[data] {len(ds)} examples loaded")

    # Optional: strip domains (e.g., vision Domain I, audio Domain P) at load time.
    # --vanilla is shorthand for --exclude-domains I,P (text-only training, no
    # vision or audio register). --exclude-domains accepts any comma-separated
    # list of single-letter domain prefixes that match _cat.startswith().
    excluded = set()
    if args.vanilla:
        excluded.update(['I', 'P'])
    if args.exclude_domains:
        excluded.update(d.strip() for d in args.exclude_domains.split(','))
    if excluded:
        before = len(ds)
        ds = ds.filter(lambda ex: not any(ex.get('_cat', '').startswith(d) for d in excluded))
        after = len(ds)
        print(f"[filter] excluded domains {sorted(excluded)}: {before} -> {after} traces (-{before-after})")

    def fmt(examples):
        out = []
        for msgs in examples["messages"]:
            # Defense in depth: drop any system message that snuck through prep
            msgs = [m for m in msgs if m.get("role") != "system"]
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,   # K0 reasons in prose, not in tagged blocks
            )
            out.append(text)
        return {"text": out}

    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)

    print()
    print("[sample] first formatted example (truncated):")
    print("-" * 60)
    print(ds[0]["text"][:1000])
    print("-" * 60)

    # Sanity: no <think> blocks should appear (K0 v2 trace gen forbids them).
    leaked_think = sum(1 for r in ds if "<think>" in r["text"])
    if leaked_think > 0:
        print(f"[FATAL] {leaked_think}/{len(ds)} formatted examples contain '<think>' tags",
              file=sys.stderr)
        print(f"[FATAL] K0 v2 must NOT train on data with thinking blocks.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[ok] 0/{len(ds)} examples contain <think> blocks")

    # Sanity: no system markers (K0 v2 dataset is NOSYS by construction).
    sys_leaked = sum(1 for r in ds if "<|im_start|>system" in r["text"])
    if sys_leaked > 0:
        print(f"[FATAL] {sys_leaked} examples contain system markers (K0 v2 is NOSYS)",
              file=sys.stderr)
        sys.exit(1)
    print(f"[ok] 0/{len(ds)} examples contain system markers")

    sft_config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_config,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    print()
    print(f"[train] starting: {args.epochs} epochs * {len(ds)} samples / "
          f"effective_batch {args.batch * args.grad_accum} = "
          f"~{(args.epochs * len(ds)) // (args.batch * args.grad_accum)} steps")
    trainer.train()

    print()
    print(f"[save] writing adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[save] adapter persisted")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, choices=[500, 1000, 2500, 5000, 7500],
                   help="If set, derives --data and --output from the tier")
    p.add_argument("--data",
                   help="Path to SFT training jsonl. Overrides --tier if both set.")
    p.add_argument("--output",
                   help="Adapter output directory. Overrides --tier if both set.")
    p.add_argument("--model", default="unsloth/Qwen3.5-9B")
    p.add_argument("--max_seq", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--vanilla", action="store_true",
                   help="Strip vision (Domain I) and audio (Domain P) at load time. "
                        "Equivalent to --exclude-domains I,P. Trains text-only K0 from any tier.")
    p.add_argument("--exclude-domains", default="",
                   help="Comma-separated domain prefixes to skip at load time, "
                        "e.g. 'I,P' to skip vision and audio. Matches _cat.startswith().")
    args = p.parse_args()

    if args.tier:
        if not args.data:
            args.data = f"dataset/v2/cumulative/tier_{args.tier}/sft_train.jsonl"
        if not args.output:
            args.output = f"adapters/k0_v2_sft_t{args.tier}"

    if not args.data:
        print("[error] --data or --tier required", file=sys.stderr)
        sys.exit(1)

    if args.skip_train:
        if not os.path.isdir(args.output):
            print(f"[error] --skip-train set but adapter dir not found: {args.output}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[skip-train] adapter at {args.output}")
        return

    print(f"[load] base model: {args.model} (vision-aware via FastVisionModel)")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    do_train(args, model, tokenizer)
    print("[done] SFT v2 stage complete (vision tower preserved, language fine-tuned)")


if __name__ == "__main__":
    main()
