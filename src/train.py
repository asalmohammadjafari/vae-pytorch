import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ConvVAE
from utils import ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def build_dataloader(dataset_name: str, data_dir: str, batch_size: int, num_workers: int):
    transform = transforms.ToTensor()
    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }
    dataset_cls = dataset_map[dataset_name]
    train_set = dataset_cls(root=data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def loss_fn(recon_x, x, mu, logvar, beta: float):
    recon = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = build_dataloader(args.dataset, args.data_dir, args.batch_size, args.num_workers)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {checkpoint['epoch']}")

    ckpt_dir = ensure_dir("checkpoints")
    n_samples = len(train_loader.dataset)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, _ in progress:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_fn(recon, x, mu, logvar, args.beta)
            loss = loss / x.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_kl = total_kl / n_samples
        print(
            f"Epoch {epoch:03d} | loss {avg_loss:.4f} | recon {avg_recon:.4f} | kl {avg_kl:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }
        if epoch % args.save_every == 0:
            save_checkpoint(state, Path(ckpt_dir) / f"epoch_{epoch}.pt")
        save_checkpoint(state, Path(ckpt_dir) / "latest.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a convolutional VAE on MNIST/Fashion-MNIST.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
