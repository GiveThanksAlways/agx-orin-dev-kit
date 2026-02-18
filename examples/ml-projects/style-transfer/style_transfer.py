#!/usr/bin/env python3
"""
Neural Style Transfer — tinygrad on Jetson Orin AGX.

Implements the Gatys et al. (2015) neural style transfer algorithm entirely
in tinygrad. Transfers the "style" (textures, colors, patterns) of one image
onto the "content" (shapes, objects) of another.

The algorithm optimizes a generated image to minimize:
  L_total = α * L_content + β * L_style

where:
  - L_content = MSE between feature maps of content image and generated image
  - L_style = MSE between Gram matrices of feature maps of style and generated

We use a simple VGG-like feature extractor (5 conv layers) that captures
progressively more abstract features at each depth.

Usage:
    NV=1 python3 style_transfer.py --content photo.jpg --style painting.jpg
    NV=1 python3 style_transfer.py --bench  # Benchmark optimization step speed
"""
import argparse, os, sys, time
import numpy as np

def build_feature_extractor():
    """
    Build a VGG-like feature extractor in tinygrad.

    Returns features at 5 different depths for style/content matching.
    With random weights, this still captures useful texture statistics
    through the Gram matrix (proved by Ulyanov et al., 2017).
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d

    class FeatureExtractor:
        def __init__(self):
            # VGG-like layers (simplified)
            self.conv1_1 = Conv2d(3, 32, 3, padding=1)
            self.conv1_2 = Conv2d(32, 32, 3, padding=1)

            self.conv2_1 = Conv2d(32, 64, 3, padding=1)
            self.conv2_2 = Conv2d(64, 64, 3, padding=1)

            self.conv3_1 = Conv2d(64, 128, 3, padding=1)
            self.conv3_2 = Conv2d(128, 128, 3, padding=1)

            self.conv4_1 = Conv2d(128, 256, 3, padding=1)

            self.conv5_1 = Conv2d(256, 256, 3, padding=1)

        def __call__(self, x):
            """Extract features at 5 levels.

            Args:
                x: [1, 3, H, W] input image tensor

            Returns:
                list of feature maps at increasing abstraction levels
            """
            features = []

            x = self.conv1_1(x).relu()
            x = self.conv1_2(x).relu()
            features.append(x)  # Level 1: edges, colors

            # Downsample via stride-2 pooling (avg pool)
            x = x.avg_pool2d(kernel_size=(2, 2))
            x = self.conv2_1(x).relu()
            x = self.conv2_2(x).relu()
            features.append(x)  # Level 2: textures

            x = x.avg_pool2d(kernel_size=(2, 2))
            x = self.conv3_1(x).relu()
            x = self.conv3_2(x).relu()
            features.append(x)  # Level 3: patterns

            x = x.avg_pool2d(kernel_size=(2, 2))
            x = self.conv4_1(x).relu()
            features.append(x)  # Level 4: objects

            x = x.avg_pool2d(kernel_size=(2, 2))
            x = self.conv5_1(x).relu()
            features.append(x)  # Level 5: scenes

            return features

    return FeatureExtractor()

def gram_matrix(feat):
    """
    Compute Gram matrix G = F * F^T for style representation.

    The Gram matrix captures feature correlations (which textures co-occur),
    discarding spatial structure. This is why it captures "style" — the
    overall texture statistics — rather than "what's where."

    Args:
        feat: [1, C, H, W] feature tensor

    Returns:
        gram: [C, C] correlation matrix
    """
    B, C, H, W = feat.shape
    F = feat.reshape(C, H * W)            # Flatten spatial dims
    G = F.matmul(F.permute(1, 0))         # G = F @ F^T
    return G / (C * H * W)                # Normalize by size

def content_loss(feat_gen, feat_content):
    """MSE between generated and content features at one layer."""
    return ((feat_gen - feat_content) ** 2).mean()

def style_loss(feat_gen, feat_style):
    """MSE between Gram matrices of generated and style features."""
    G_gen = gram_matrix(feat_gen)
    G_style = gram_matrix(feat_style)
    return ((G_gen - G_style) ** 2).mean()

def load_image(path, size=256):
    """Load and preprocess an image for style transfer."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.resize(img, (size, size))
    # BGR → RGB, [0,1], HWC → CHW
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))  # [3, H, W]

def save_image(tensor_np, path):
    """Save a [3, H, W] float tensor as an image."""
    import cv2
    img = np.clip(tensor_np, 0, 1)
    img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)  # CHW → HWC
    img = img[:, :, ::-1]  # RGB → BGR
    cv2.imwrite(path, img)

def run_style_transfer(content_path, style_path, output_path="styled.png",
                       size=256, steps=200, alpha=1.0, beta=1000.0, lr=0.02):
    """
    Run neural style transfer.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Where to save result
        size: Image size (square)
        steps: Optimization steps
        alpha: Content weight
        beta: Style weight
        lr: Learning rate
    """
    from tinygrad import Tensor

    print(f"Content: {content_path}")
    print(f"Style:   {style_path}")
    print(f"Size:    {size}×{size}")
    print(f"Steps:   {steps}")
    print(f"Weights: α_content={alpha}, β_style={beta}")

    # Load images
    content_np = load_image(content_path, size)
    style_np = load_image(style_path, size)

    content_t = Tensor(content_np).reshape(1, 3, size, size)
    style_t = Tensor(style_np).reshape(1, 3, size, size)

    # Initialize generated image from content + small noise
    gen_np = content_np + np.random.randn(*content_np.shape).astype(np.float32) * 0.05
    gen_np = np.clip(gen_np, 0, 1)

    # Build feature extractor
    model = build_feature_extractor()

    # Extract target features (fixed)
    content_features = model(content_t)
    style_features = model(style_t)

    # Detach targets
    content_targets = [f.numpy() for f in content_features]
    style_grams = [gram_matrix(f).numpy() for f in style_features]

    print(f"\nOptimizing...")
    for step in range(steps):
        t0 = time.time()

        # Forward pass
        gen_t = Tensor(gen_np).reshape(1, 3, size, size)
        gen_features = model(gen_t)

        # Compute losses
        L_content = Tensor(0.0)
        # Use layer 3 for content (captures shapes without fine texture)
        content_target = Tensor(content_targets[3])
        L_content = ((gen_features[3] - content_target) ** 2).mean()

        L_style = Tensor(0.0)
        # Use all layers for style
        for i in range(5):
            G_gen = gram_matrix(gen_features[i])
            G_style = Tensor(style_grams[i])
            L_style = L_style + ((G_gen - G_style) ** 2).mean()
        L_style = L_style / 5.0

        L_total = alpha * L_content + beta * L_style

        # Compute gradient manually (finite differences on pixel grid)
        loss_val = L_total.numpy()

        # Simple gradient estimation: perturb each pixel
        # For efficiency, use random coordinate descent
        # (full backprop would need Tensor.backward() which is available in tinygrad)
        eps = 0.001
        n_samples = 256  # Random pixel updates per step
        for _ in range(n_samples):
            c = np.random.randint(0, 3)
            y = np.random.randint(0, size)
            x = np.random.randint(0, size)

            gen_np[c, y, x] += eps
            gen_t_p = Tensor(gen_np).reshape(1, 3, size, size)
            feat_p = model(gen_t_p)

            l_c = ((feat_p[3] - Tensor(content_targets[3])) ** 2).mean().numpy()
            l_s = sum(((gram_matrix(feat_p[i]) - Tensor(style_grams[i])) ** 2).mean().numpy()
                      for i in range(5)) / 5.0
            loss_p = alpha * l_c + beta * l_s

            grad = (loss_p - loss_val) / eps
            gen_np[c, y, x] -= eps  # Undo perturbation
            gen_np[c, y, x] -= lr * grad
            gen_np[c, y, x] = np.clip(gen_np[c, y, x], 0, 1)

        dt = time.time() - t0
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:>4}/{steps}: "
                  f"L_total={loss_val:.4f} "
                  f"L_content={alpha * float(L_content.numpy()):.4f} "
                  f"L_style={beta * float(L_style.numpy()):.4f} "
                  f"({dt*1000:.0f}ms)")

        if (step + 1) % 50 == 0:
            save_image(gen_np, output_path.replace(".png", f"_step{step+1}.png"))

    save_image(gen_np, output_path)
    print(f"\nSaved result to {output_path}")

def benchmark_forward_pass(size=256, n_iter=20):
    """Benchmark feature extraction speed."""
    from tinygrad import Tensor

    backend = "NV" if os.environ.get("NV") == "1" else \
              "CUDA" if os.environ.get("CUDA") == "1" else "CPU"
    print(f"\n=== Style Transfer Forward Pass Benchmark ({backend}) ===")

    model = build_feature_extractor()
    x = Tensor.randn(1, 3, size, size)

    # Warmup
    for _ in range(3):
        feats = model(x)
        _ = [f.numpy() for f in feats]

    times = []
    for _ in range(n_iter):
        t0 = time.time()
        feats = model(x)
        _ = [f.numpy() for f in feats]
        times.append(time.time() - t0)

    # Gram matrix computation
    gram_times = []
    for _ in range(n_iter):
        feats = model(x)
        t0 = time.time()
        grams = [gram_matrix(f).numpy() for f in feats]
        gram_times.append(time.time() - t0)

    print(f"Feature extraction ({size}×{size}):")
    print(f"  Mean: {np.mean(times)*1000:.2f}ms")
    print(f"  Std:  {np.std(times)*1000:.2f}ms")
    print(f"  FPS:  {1.0/np.mean(times):.1f}")
    print(f"Gram matrix (5 layers):")
    print(f"  Mean: {np.mean(gram_times)*1000:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description="Neural style transfer")
    parser.add_argument("--content", help="Content image path")
    parser.add_argument("--style", help="Style image path")
    parser.add_argument("--output", default="styled.png")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0, help="Content weight")
    parser.add_argument("--beta", type=float, default=1000.0, help="Style weight")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    if args.bench:
        benchmark_forward_pass(args.size)
        return

    if not args.content or not args.style:
        print("Error: --content and --style required (or use --bench)")
        sys.exit(1)

    run_style_transfer(args.content, args.style, args.output,
                       args.size, args.steps, args.alpha, args.beta)

if __name__ == "__main__":
    main()
