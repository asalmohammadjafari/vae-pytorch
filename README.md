# vae-pytorch

Simple convolutional Variational Autoencoder (VAE) in PyTorch for image reconstruction and latent representation learning on MNIST or Fashion-MNIST.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Train

```bash
python src/train.py --dataset mnist --epochs 15 --batch-size 128
```

Checkpoints are saved in `checkpoints/`.

## 🖼️ Generate / Reconstruct

```bash
python src/generate.py --checkpoint checkpoints/latest.pt --mode both
```

Outputs are saved in `outputs/`.

## 🗂️ Project Structure

```text
vae-pytorch/
  README.md
  requirements.txt
  src/
    model.py
    train.py
    generate.py
    utils.py
```

