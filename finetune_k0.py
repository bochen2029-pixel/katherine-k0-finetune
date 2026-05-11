"""
DEPRECATED v1 LEGACY (2026-05-10).

This is the v1 SFT trainer. Preserved for historical reference and the v1
training run. Uses `unsloth.FastModel` which silently DROPS the vision tower
from the merged GGUF. DO NOT USE for v2 training — produces vision-broken model.

For v2 SFT use scripts/finetune_k0_v2.py (FastVisionModel, vision-preserving).

---

Original v1 docstring:

Stage 1 — QLoRA SFT trainer for Katherine k0 on Qwen3.5-9B.

Hyperparameters (from CLOUD.md derivation):
  rank       = 64        (alpha = 128, dropout = 0.05)
  epochs     = 3
  lr         = 1e-4 (cosine, warmup 5%)
  batch      = 16 per device, grad_accum = 2 (effective 32)
  max_seq    = 1024      (data p99 = 246 tokens; 4× margin)
  bf16       = on        (H200/H100 native)
  optim      = adamw_8bit
  thinking   = OFF       (K0 doesn't reason out loud)
  sys-prompt = STRIPPED  (handled at preprocess time by prep_dataset.py)

Failsafe:
  - Adapter saved per epoch; final adapter saved to --output
  - Use --skip-train + --output <existing_adapter_dir> to load + retry GGUF only
  - Atomic JSONL training records; no partial writes
"""
import argparse
import os
import sys

# Unsloth MUST import before transformers/peft/trl
from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


def do_train(args, model, tokenizer):
    """Attach LoRA, format dataset, train, save adapter."""
    model = FastModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"[data] loading {args.data}")
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"[data] {len(ds)} examples loaded")

    def fmt(examples):
        out = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,   # K0 reasons in prose, not in tagged blocks
            )
            out.append(text)
        return {"text": out}

    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)

    # Sanity check first formatted example
    print()
    print("[sample] first formatted example (truncated):")
    print("-" * 60)
    print(ds[0]["text"][:1000])
    print("-" * 60)

    # Confirm no <think> blocks leaked in (they shouldn't with enable_thinking=False
    # and no <think> in source data)
    leaked = sum(1 for r in ds if "<think>" in r["text"])
    if leaked > 0:
        print(f"[warn] {leaked}/{len(ds)} formatted examples contain '<think>' tags")
        print(f"[warn] K0 should NOT have thinking blocks — investigate the source data")
        # Don't abort — just warn

    # Confirm system prompts are gone from formatted text (rough check)
    sys_leaked = sum(1 for r in ds if "system" in r["text"][:200].lower() and "Katherine Hale" in r["text"][:500])
    if sys_leaked > 0:
        print(f"[warn] {sys_leaked} examples appear to still have system content; verify prep_dataset.py ran cleanly")

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
    print(f"[train] starting: {args.epochs} epochs × {len(ds)} samples / "
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
    p.add_argument("--data", default="data/k0_canonical.jsonl")
    p.add_argument("--output", default="adapters/k0_sft_adapter")
    p.add_argument("--model", default="unsloth/Qwen3.5-9B")
    p.add_argument("--max_seq", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; assume adapter already exists at --output. "
                        "Useful for re-running merge/GGUF stages alone.")
    args = p.parse_args()

    if args.skip_train:
        if not os.path.isdir(args.output):
            print(f"[error] --skip-train set but adapter dir not found: {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"[skip-train] adapter assumed at {args.output}; nothing to do here")
        return

    print(f"[load] base model: {args.model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    do_train(args, model, tokenizer)
    print("[done] SFT stage complete")


if __name__ == "__main__":
    main()
