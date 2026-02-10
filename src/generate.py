import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import ConvVAE
from utils import ensure_dir, get_device, load_checkpoint


def build_test_loader(dataset_name: str, data_dir: str, batch_size: int):
    transform = transforms.ToTensor()
    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }
    dataset_cls = dataset_map[dataset_name]
    test_set = dataset_cls(root=data_dir, train=False, transform=transform, download=True)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and reconstruct images from a trained VAE.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default="both", choices=["sample", "reconstruct", "both"])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-recon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=None)
    return parser.parse_args()


def main(args):
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device)

    latent_dim = args.latent_dim
    if latent_dim is None:
        latent_dim = checkpoint.get("args", {}).get("latent_dim", 20)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = ensure_dir("outputs")

    with torch.no_grad():
        if args.mode in ("sample", "both"):
            samples = model.sample(args.num_samples, device)
            save_image(samples.cpu(), Path(out_dir) / "samples.png", nrow=8)
            print(f"Saved samples: {Path(out_dir) / 'samples.png'}")

        if args.mode in ("reconstruct", "both"):
            test_loader = build_test_loader(args.dataset, args.data_dir, args.batch_size)
            x, _ = next(iter(test_loader))
            x = x[: args.num_recon].to(device)
            recon, _, _ = model(x)
            comparison = torch.cat([x, recon], dim=0).cpu()
            save_image(comparison, Path(out_dir) / "reconstruction.png", nrow=args.num_recon)
            print(f"Saved reconstructions: {Path(out_dir) / 'reconstruction.png'}")


if __name__ == "__main__":
    main(parse_args())
