import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import wandb

from models import AE  # uses your AE class


def get_dataloaders(batch_size, data_dir="./data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    val_ds   = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def plot_latent_space(model, loader, device, max_points=10000):
    """Return a Matplotlib figure of the 2D latent space colored by labels."""
    model.eval()
    zs, ys = [], []
    collected = 0
    for x, y in loader:
        x = x.to(device).view(x.size(0), -1)
        z = model.encode(x)  # [B, latent_dim]
        zs.append(z.detach().cpu())
        ys.append(y.detach().cpu())
        collected += x.size(0)
        if collected >= max_points:
            break

    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    if Z.size(1) != 2:
        raise ValueError(f"Latent space visualization requires latent_dim=2, got {Z.size(1)}")

    fig = plt.figure(figsize=(6, 5), dpi=120)
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=Y.numpy(), s=6, alpha=0.7, cmap="tab10")
    plt.title("AE Latent Space (2D)")
    plt.xlabel("z1"); plt.ylabel("z2")
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit label")
    plt.tight_layout()
    return fig


def bce_loss(recon, x):
    # recon: [B, 784], x: [B, 784], both in [0,1]
    # Use binary cross entropy loss for each pixel
    return nn.functional.binary_cross_entropy(recon, x, reduction="mean")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)
        optimizer.zero_grad(set_to_none=True)
        recon, z = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)
        recon, z = model(x)
        loss = criterion(recon, x)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--project", type=str, default="mnist-ae")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(args.batch_size, args.data_dir)

    # Model
    input_dim = 28 * 28  # MNIST
    model = AE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)

    # Optim + Loss (now using BCE for binary images)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = bce_loss

    # Weights & Biases
    wandb.init(project=args.project, entity=args.entity,
               config={
                   "epochs": args.epochs,
                   "name": "AE",
                   "batch_size": args.batch_size,
                   "lr": args.lr,
                   "latent_dim": args.latent_dim,
                   "optimizer": "Adam",
                   "criterion": "BCELoss",
               })
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Log losses every epoch
        wandb.log({
            "epoch": epoch,
            "loss/train_recon": train_loss,
            "loss/val_recon": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Every 5 epochs: visualize latent space (requires latent_dim=2)
        if epoch % 5 == 0 and args.latent_dim == 2:
            fig = plot_latent_space(model, val_loader, device)
            wandb.log({"plots/latent_space": wandb.Image(fig), "epoch_plot": epoch})
            plt.close(fig)

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"ae_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": wandb.config,
        }, ckpt_path)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
