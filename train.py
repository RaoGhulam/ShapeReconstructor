import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from render import render_shape
from torchvision import transforms
import matplotlib.pyplot as plt

from model import (
    ShapeReconstructor, GRID_SIZE, NUM_SHAPES,
    SHAPE_NAMES, SHAPE_PARAM_COUNTS, MAX_PARAMS
)
# from utils.dataset import ShapeDataset, collate_fn

# ─────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────

ce_loss = nn.CrossEntropyLoss()


def compute_loss(
    model: ShapeReconstructor,
    images: torch.Tensor,       # (B, 1, H, W)  – target image
    commands: list,             # list of B command-sequences
    device: torch.device,
) -> torch.Tensor:
    """
    Teacher-forced loss over a batch.

    For each sample we iterate over its commands:
      - Predict t1 (shape type)
      - For each parameter step k: predict t_{k+1} given true t1..tk
    """
    B = images.shape[0]
    images = images.to(device)
    canvas = torch.zeros_like(images)   # start with blank canvas

    total_loss = torch.zeros((), device=device)
    total_steps = 0

    for b in range(B):
        target_b = images[b].unsqueeze(0)
        canvas_b = canvas[b].unsqueeze(0)

        for cmd in commands[b]:
            shape_name = cmd["shape"]
            shape_idx  = torch.tensor([cmd["shape_idx"]], dtype=torch.long, device=device)
            params     = cmd["params"]
            params_t = torch.tensor(params, dtype=torch.float32, device=device)
            params_t = torch.round(params_t / 255 * 15).long()

            # ── t1: shape prediction ──
            out = model(
                target_b, canvas_b,
                shape_idx=None,
                prev_param_tokens=torch.zeros(1, 0, dtype=torch.long, device=device),
                step=0,
            )

            loss = ce_loss(out["cmd_logits"], shape_idx)
            total_loss += loss
            total_steps += 1

            # ── parameters (teacher forcing) ──
            n_params = SHAPE_PARAM_COUNTS[shape_name]

            for k in range(n_params):
                prev_p = params_t[:k].unsqueeze(0)

                out = model(
                    target_b, canvas_b,
                    shape_idx=shape_idx,
                    prev_param_tokens=prev_p,
                    step=k+1,
                )

                if out["param_logits"] is None:
                    raise RuntimeError(
                        f"Model predicted STOP during param step {k} for shape {shape_name}"
                    )

                target_tok = params_t[k].unsqueeze(0)
                loss = ce_loss(out["param_logits"], target_tok)

                total_loss += loss
                total_steps += 1

            # ── update canvas with GT ──
            if shape_name!="stop":
              gt_shape = render_shape(shape_name, params, device=device)
              canvas_b = canvas_b + gt_shape.detach()

    return total_loss / total_steps


# ─────────────────────────────────────────────
#  Train / Eval epoch
# ─────────────────────────────────────────────

def run_epoch(model, loader, optimiser, device, train: bool) -> float:
    model.train(train)
    epoch_loss = 0.0

    with torch.set_grad_enabled(train):
          for images, targets in loader:

            loss = compute_loss(model, images, targets, device)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ── Dataset ──
    transform = transforms.ToTensor()
    dataset = CustomDataset(args.dataset_path, transform=transform)

    # Split into train/val (10% validation)
    n_val  = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    # Default split
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_indices = train_ds.indices
    val_indices = val_ds.indices

    # ── Model ──
    model = ShapeReconstructor(img_size=args.img_size).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model parameters: {num_params:,}")

    # ── Optimiser + Scheduler ──
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if args.resume and os.path.isfile(args.checkpoint_path):
        print(f"[train] Loading checkpoint from {args.checkpoint_path}")
        
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

        train_indices = checkpoint["train_indices"]
        val_indices   = checkpoint["val_indices"]

        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds   = torch.utils.data.Subset(dataset, val_indices)
        
        print(f"[train] Resumed from epoch {start_epoch+1}")
        print(f"[train] Previous best val_loss: {best_val_loss:.6f}")

    # Custom collate function
    def custom_collate(batch):
        images, shapes_list = zip(*batch)  # unzip
        images = torch.stack(images)        # batch images normally
        return images, shapes_list          # keep shapes as list of lists
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,  
        collate_fn=custom_collate,  # handles variable shapes
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=0
    )

    print(f"[train] dataset size: {len(dataset)}, train: {len(train_ds)}, val: {len(val_ds)}")

    # ── Checkpoint dir ──
    os.makedirs(args.save_dir, exist_ok=True)
    epochs_no_improve = 0
    early_stop_patience = 5

    # ── Training loop ──
    print(f"[train] starting training for {args.epochs} epochs …\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimiser, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimiser, device, train=False)
        elapsed    = time.time() - t0

        scheduler.step(val_loss)

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:3d}/{start_epoch+args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimiser.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Check if validation improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # reset counter
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":   val_loss,
                "args":       vars(args),
                "train_losses": train_losses,
                "val_losses": val_losses, 
                "train_indices": train_ds.indices,
                "val_indices": val_ds.indices,
            }, ckpt_path)
            print(f"  ✓ saved best model → {ckpt_path}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[train] Early stopping triggered at epoch {epoch}")
            break

    # ── Training done ──
    print(f"\n[train] done. Best val_loss = {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Data config
    parser.add_argument("--dataset_path", type=str, default="dataset")

    # Model config
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_shapes", type=int, default=4)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    # Optional: print config
    print("[config]", vars(args))

    main(args)