"""
Clipper — Training Script

Only the 4 custom modules are trained (~3M parameters).
CLIP backbone stays completely frozen.

Features:
  - Gradient accumulation  → large effective batch on small GPU
  - FP16 mixed precision   → halves VRAM usage
  - Linear warmup + cosine decay LR schedule
  - Saves best checkpoint by R@1

Usage:
    python train.py
"""
import os
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data         import DataLoader
from torch.cuda.amp           import GradScaler, autocast
from torch.optim              import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config   import ClipperConfig
from model    import ClipperModel
from dataset  import VideoTextDataset, collate_fn
from evaluate import run_evaluation


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(config: ClipperConfig = None):
    config = config or ClipperConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  Clipper Training")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {config.num_epochs}")
    print(f"  Batch     : {config.batch_size} × {config.accum_steps} accum"
          f"  = {config.batch_size * config.accum_steps} effective")
    print(f"  FP16      : {config.fp16}")
    print(f"  LR        : {config.lr}")
    print(f"{'='*55}\n")

    # ── Model ────────────────────────────────────────────────────
    model = ClipperModel(config).to(device)
    model.config.use_custom_modules = True

    if os.path.exists(config.weights_path):
        ckpt = torch.load(config.weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {config.weights_path}")

    # ── Datasets ─────────────────────────────────────────────────
    train_ds = VideoTextDataset(
        config.train_ann_path, config,
        model.preprocess, model.tokenizer, split="train"
    )
    val_ds = VideoTextDataset(
        config.val_ann_path, config,
        model.preprocess, model.tokenizer, split="val"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 16,
        shuffle     = False,
        num_workers = config.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True
    )

    # ── Optimizer — trainable params only ─────────────────────────
    trainable   = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = AdamW(
        trainable,
        lr           = config.lr,
        weight_decay = config.weight_decay,
        betas        = (0.9, 0.98),
        eps          = 1e-6
    )
    total_steps = (
        len(train_loader) // config.accum_steps
    ) * config.num_epochs
    scheduler   = get_cosine_schedule(optimizer, config.warmup_steps,
                                       total_steps)
    scaler      = GradScaler(enabled=config.fp16)

    os.makedirs(os.path.dirname(config.weights_path), exist_ok=True)

    best_r1     = 0.0
    global_step = 0
    history     = []

    # ── Training Loop ────────────────────────────────────────────
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss  = 0.0
        accum_count = 0
        t0          = time.time()

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            frames = batch["frames"].to(device)    # [B, T, C, H, W]
            tokens = batch["tokens"].to(device)    # [B, 77]

            with autocast(enabled=config.fp16):
                out  = model(frames, tokens)
                loss = out["loss"] / config.accum_steps

            scaler.scale(loss).backward()
            epoch_loss  += loss.item() * config.accum_steps
            accum_count += 1

            if accum_count == config.accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count  = 0
                global_step += 1

                if global_step % 50 == 0:
                    avg = epoch_loss / (step + 1)
                    lr  = scheduler.get_last_lr()[0]
                    print(f"  E{epoch} | step {global_step:4d} | "
                          f"loss {avg:.4f} | lr {lr:.2e} | "
                          f"τ {out['logit_scale']:.2f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed  = time.time() - t0
        print(f"\nEpoch {epoch}/{config.num_epochs} — "
              f"loss: {avg_loss:.4f}  time: {elapsed:.0f}s")

        # ── Evaluation ───────────────────────────────────────────
        if epoch % config.eval_every == 0:
            metrics = run_evaluation(model, val_loader, device, config)
            r1 = metrics["R@1"]
            print(f"  Val  R@1: {r1:.1f}  R@5: {metrics['R@5']:.1f}  "
                  f"R@10: {metrics['R@10']:.1f}  "
                  f"MdR: {metrics['MdR']:.1f}")

            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), config.best_ckpt_path)
                print(f"  ✅ New best R@1={r1:.1f} "
                      f"→ saved {config.best_ckpt_path}")

        # Save latest checkpoint
        torch.save(model.state_dict(), config.weights_path)

    json.dump(history,
              open("data/training_history.json", "w"), indent=2)
    print(f"\nTraining complete. Best R@1: {best_r1:.1f}")
    print(f"Best model saved → {config.best_ckpt_path}")


if __name__ == "__main__":
    train()
