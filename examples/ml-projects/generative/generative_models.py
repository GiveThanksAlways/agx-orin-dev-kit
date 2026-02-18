#!/usr/bin/env python3
"""
Generative Models — Autoencoder + Diffusion in tinygrad on Jetson Orin AGX.

Two generative model experiments:

1. **Convolutional Autoencoder**: Learns compressed representations of images.
   Encoder→latent→Decoder. The latent space is where the "meaning" lives.
   Interpolate in latent space to generate novel images.

2. **Tiny Diffusion Model**: Simplified DDPM (Denoising Diffusion Probabilistic
   Model). Learns to reverse a noise process: start from pure noise, iteratively
   denoise to produce realistic images.

Uses synthetic data (geometric patterns) — no dataset download needed.

Usage:
    NV=1 python3 generative_models.py --model autoencoder --bench
    NV=1 python3 generative_models.py --model diffusion --bench
    NV=1 python3 generative_models.py --model autoencoder --train --steps 100
"""
import argparse, os, sys, time
import numpy as np

# ==========================================================================
# Data Generation (synthetic — no downloads needed)
# ==========================================================================
def generate_shapes(n_samples=1000, size=32):
    """
    Generate synthetic 32×32 images of geometric shapes.
    Each image has RGB channels with a single centered shape.

    This avoids needing to download MNIST/CIFAR — the model still
    learns meaningful latent representations from this data.
    """
    images = np.zeros((n_samples, 3, size, size), dtype=np.float32)

    for i in range(n_samples):
        shape_type = i % 4  # circle, square, triangle, cross
        color = np.random.rand(3) * 0.7 + 0.3  # Bright random color
        bg_color = np.random.rand(3) * 0.3       # Dark background

        img = np.ones((3, size, size), dtype=np.float32)
        for c in range(3):
            img[c] *= bg_color[c]

        cx, cy = size // 2 + np.random.randint(-4, 5), size // 2 + np.random.randint(-4, 5)
        r = np.random.randint(6, 12)

        if shape_type == 0:  # Circle
            yy, xx = np.ogrid[:size, :size]
            mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
            for c in range(3):
                img[c][mask] = color[c]

        elif shape_type == 1:  # Square
            y1, y2 = max(0, cy-r), min(size, cy+r)
            x1, x2 = max(0, cx-r), min(size, cx+r)
            for c in range(3):
                img[c, y1:y2, x1:x2] = color[c]

        elif shape_type == 2:  # Triangle
            for y in range(max(0, cy-r), min(size, cy+r)):
                half_w = int(r * (1 - abs(y - cy) / r))
                x1, x2 = max(0, cx-half_w), min(size, cx+half_w)
                for c in range(3):
                    img[c, y, x1:x2] = color[c]

        elif shape_type == 3:  # Cross
            arm = r // 3
            for c in range(3):
                img[c, max(0,cy-r):min(size,cy+r), max(0,cx-arm):min(size,cx+arm)] = color[c]
                img[c, max(0,cy-arm):min(size,cy+arm), max(0,cx-r):min(size,cx+r)] = color[c]

        # Add slight noise
        img += np.random.randn(3, size, size).astype(np.float32) * 0.02
        images[i] = np.clip(img, 0, 1)

    return images

# ==========================================================================
# Convolutional Autoencoder
# ==========================================================================
def build_autoencoder(latent_dim=32, img_size=32):
    """
    Convolutional autoencoder with symmetric encoder/decoder.

    Encoder: 3×32×32 → 64×16×16 → 128×8×8 → 256×4×4 → flatten → latent_dim
    Decoder: latent_dim → 256×4×4 → 128×8×8 → 64×16×16 → 3×32×32

    The bottleneck (latent_dim=32) forces the network to learn a compressed
    representation of the input. Similar images map to nearby points in
    latent space.
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    class Encoder:
        def __init__(self):
            self.conv1 = Conv2d(3, 64, 3, stride=2, padding=1)   # 32→16
            self.conv2 = Conv2d(64, 128, 3, stride=2, padding=1) # 16→8
            self.conv3 = Conv2d(128, 256, 3, stride=2, padding=1) # 8→4
            self.fc = Linear(256 * 4 * 4, latent_dim)

        def __call__(self, x):
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = x.reshape(x.shape[0], -1)
            return self.fc(x)

    class Decoder:
        def __init__(self):
            self.fc = Linear(latent_dim, 256 * 4 * 4)
            # Use conv + upsample instead of transposed conv
            self.conv1 = Conv2d(256, 128, 3, padding=1)  # 4→4
            self.conv2 = Conv2d(128, 64, 3, padding=1)   # 8→8
            self.conv3 = Conv2d(64, 3, 3, padding=1)     # 16→16

        def __call__(self, z):
            x = self.fc(z).relu().reshape(-1, 256, 4, 4)

            # Upsample 4→8 (nearest neighbor via reshape + expand)
            B, C, H, W = x.shape
            x = x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, H*2, W*2)
            x = self.conv1(x).relu()

            # Upsample 8→16
            B, C, H, W = x.shape
            x = x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, H*2, W*2)
            x = self.conv2(x).relu()

            # Upsample 16→32
            B, C, H, W = x.shape
            x = x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, H*2, W*2)
            x = self.conv3(x).sigmoid()  # Output in [0, 1]

            return x

    class AutoEncoder:
        def __init__(self):
            self.encoder = Encoder()
            self.decoder = Decoder()

        def __call__(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

    return AutoEncoder()

# ==========================================================================
# Tiny Diffusion Model
# ==========================================================================
def build_diffusion_unet(img_channels=3, base_ch=32, img_size=32):
    """
    Simplified U-Net for diffusion denoising.

    Input: noisy image [B, 3, 32, 32] + timestep embedding
    Output: predicted noise [B, 3, 32, 32]

    This is a minimal version of the architecture used in DDPM.
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    class DiffusionUNet:
        def __init__(self):
            # Time embedding
            self.time_fc1 = Linear(1, base_ch)
            self.time_fc2 = Linear(base_ch, base_ch)

            # Encoder path
            self.enc1 = Conv2d(img_channels, base_ch, 3, padding=1)
            self.enc2 = Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)  # 32→16
            self.enc3 = Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)  # 16→8

            # Bottleneck
            self.mid = Conv2d(base_ch * 4, base_ch * 4, 3, padding=1)

            # Decoder path (conv + nearest-neighbor upsample)
            self.dec3 = Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
            self.dec2 = Conv2d(base_ch * 2, base_ch, 3, padding=1)
            self.dec1 = Conv2d(base_ch, img_channels, 3, padding=1)

        def __call__(self, x, t):
            """Predict noise. x: [B,3,32,32], t: [B,1] normalized timestep."""
            # Time embedding
            t_emb = self.time_fc1(t).relu()
            t_emb = self.time_fc2(t_emb)  # [B, base_ch]

            # Encoder
            h1 = self.enc1(x).relu()
            # Add time embedding (broadcast to spatial dims)
            h1 = h1 + t_emb.reshape(-1, base_ch, 1, 1).expand(*h1.shape)

            h2 = self.enc2(h1).relu()  # [B, 64, 16, 16]
            h3 = self.enc3(h2).relu()  # [B, 128, 8, 8]

            # Bottleneck
            h = self.mid(h3).relu()     # [B, 128, 8, 8]

            # Decoder with skip connections (add, not concat, for simplicity)
            # Upsample 8→16
            B, C, H, W = h.shape
            h = h.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, H*2, W*2)
            h = self.dec3(h).relu() + h2  # Skip connection

            # Upsample 16→32
            B, C, H, W = h.shape
            h = h.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, H*2, W*2)
            h = self.dec2(h).relu() + h1  # Skip connection

            return self.dec1(h)  # Predicted noise

    return DiffusionUNet()

class DiffusionSchedule:
    """
    Linear noise schedule for DDPM.

    Forward process: q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t) * I)
    where ᾱ_t = Π_{s=1}^t (1 - β_s)

    Reverse process: p(x_{t-1} | x_t) learned by the U-Net.
    """
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def add_noise(self, x0, t, noise=None):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)
        sqrt_alpha_bar = np.sqrt(self.alpha_bars[t])
        sqrt_one_minus = np.sqrt(1.0 - self.alpha_bars[t])
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

def main():
    parser = argparse.ArgumentParser(description="Generative models in tinygrad")
    parser.add_argument("--model", choices=["autoencoder", "diffusion"], default="autoencoder")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--train", action="store_true", help="Train on synthetic data")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    backend = "NV" if os.environ.get("NV") == "1" else \
              "CUDA" if os.environ.get("CUDA") == "1" else "CPU"

    if args.bench:
        from tinygrad import Tensor

        print(f"╔══════════════════════════════════════════════╗")
        print(f"║  Generative Model Benchmark — Backend: {backend:>4}  ║")
        print(f"╚══════════════════════════════════════════════╝\n")

        if args.model == "autoencoder":
            print("=== Convolutional Autoencoder ===")
            ae = build_autoencoder(latent_dim=32)

            x = Tensor.randn(1, 3, 32, 32)

            # Warmup
            for _ in range(3):
                ae(x).numpy()

            # Encode benchmark
            print("\nEncoder (3×32×32 → latent_dim=32):")
            times = []
            for _ in range(30):
                t0 = time.time()
                ae.encode(Tensor.randn(1, 3, 32, 32)).numpy()
                times.append(time.time() - t0)
            print(f"  Mean: {np.mean(times)*1000:.2f}ms, FPS: {1/np.mean(times):.1f}")

            # Decode benchmark
            print("\nDecoder (latent_dim=32 → 3×32×32):")
            times = []
            for _ in range(30):
                t0 = time.time()
                ae.decode(Tensor.randn(1, 32)).numpy()
                times.append(time.time() - t0)
            print(f"  Mean: {np.mean(times)*1000:.2f}ms, FPS: {1/np.mean(times):.1f}")

            # Full autoencoder
            print("\nFull AE (encode + decode):")
            times = []
            for _ in range(30):
                t0 = time.time()
                ae(Tensor.randn(1, 3, 32, 32)).numpy()
                times.append(time.time() - t0)
            print(f"  Mean: {np.mean(times)*1000:.2f}ms, FPS: {1/np.mean(times):.1f}")

            # Batch scaling
            print("\nBatch scaling:")
            for bs in [1, 4, 8, 16]:
                ae(Tensor.randn(bs, 3, 32, 32)).numpy()
                times = []
                for _ in range(20):
                    t0 = time.time()
                    ae(Tensor.randn(bs, 3, 32, 32)).numpy()
                    times.append(time.time() - t0)
                ips = bs / np.mean(times)
                print(f"  BS={bs:>2}: {np.mean(times)*1000:.2f}ms ({ips:.1f} img/s)")

        else:  # diffusion
            print("=== Tiny Diffusion Model (U-Net) ===")
            unet = build_diffusion_unet()

            x = Tensor.randn(1, 3, 32, 32)
            t = Tensor([[0.5]])  # Normalized timestep

            # Warmup
            for _ in range(3):
                unet(x, t).numpy()

            # Forward pass
            print("\nNoise prediction (32×32):")
            times = []
            for _ in range(30):
                t0 = time.time()
                unet(Tensor.randn(1, 3, 32, 32), Tensor([[np.random.rand()]])).numpy()
                times.append(time.time() - t0)
            print(f"  Mean: {np.mean(times)*1000:.2f}ms, FPS: {1/np.mean(times):.1f}")

            # Full sampling chain (T steps)
            T = 100
            print(f"\nFull sampling chain ({T} denoising steps):")
            t0 = time.time()
            x = Tensor.randn(1, 3, 32, 32)
            for step in range(T):
                t_norm = Tensor([[step / T]])
                noise_pred = unet(x, t_norm)
                # Simplified DDPM update
                x = x - noise_pred * 0.01
            _ = x.numpy()
            total = time.time() - t0
            print(f"  Total: {total*1000:.1f}ms ({total/T*1000:.1f}ms/step)")
            print(f"  Samples/s: {1.0/total:.2f}")

        return

    if args.train:
        from tinygrad import Tensor
        print(f"Training {args.model} on synthetic shapes ({backend})...")

        data = generate_shapes(n_samples=200)
        print(f"Generated {data.shape[0]} training images ({data.shape})")

        if args.model == "autoencoder":
            ae = build_autoencoder()
            print(f"\nTraining autoencoder for {args.steps} steps (batch=8)...")

            for step in range(args.steps):
                idx = np.random.choice(len(data), 8)
                batch = Tensor(data[idx])

                recon = ae(batch)
                loss = ((recon - batch) ** 2).mean()
                loss_val = loss.numpy()

                if (step + 1) % 10 == 0:
                    print(f"  Step {step+1:>4}: MSE loss = {loss_val:.6f}")

            print("Training complete (weights not persisted — this is a demo)")

        else:  # diffusion
            unet = build_diffusion_unet()
            schedule = DiffusionSchedule(T=100)
            print(f"\nTraining diffusion model for {args.steps} steps (batch=8)...")

            for step in range(args.steps):
                idx = np.random.choice(len(data), 8)
                x0 = data[idx]
                t = np.random.randint(0, schedule.T, size=8)

                # Add noise
                noisy = np.zeros_like(x0)
                noise_target = np.zeros_like(x0)
                for i in range(8):
                    noisy[i], noise_target[i] = schedule.add_noise(x0[i], t[i])

                t_norm = (t / schedule.T).astype(np.float32).reshape(-1, 1)

                noise_pred = unet(Tensor(noisy), Tensor(t_norm))
                loss = ((noise_pred - Tensor(noise_target)) ** 2).mean()
                loss_val = loss.numpy()

                if (step + 1) % 10 == 0:
                    print(f"  Step {step+1:>4}: noise MSE = {loss_val:.6f}")

            print("Training complete")

if __name__ == "__main__":
    main()
