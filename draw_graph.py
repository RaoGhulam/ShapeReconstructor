import torch
import os
import argparse
import matplotlib.pyplot as plt

def draw_graph(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])

    # ── Plot train/val loss curves ──
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(args.save_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"[train] Loss curve saved → {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/content")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    draw_graph(args)