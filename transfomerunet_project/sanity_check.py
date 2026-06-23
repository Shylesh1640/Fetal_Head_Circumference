"""
sanity_check.py
Run this FIRST, before train.py, to catch configuration/data issues early.
Checks: package versions, CUDA availability, dataset path, mask filling
correctness, and one forward/backward pass through the real model.

Usage:
    python sanity_check.py
"""

import sys
import os


def check_imports():
    print("== Checking imports ==")
    try:
        import torch
        import segmentation_models_pytorch as smp
        import timm
        import albumentations as A
        import cv2
        print(f"torch: {torch.__version__}")
        print(f"segmentation_models_pytorch: {smp.__version__}")
        print(f"timm: {timm.__version__}")
        print(f"albumentations: {A.__version__}")
        print(f"opencv: {cv2.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("OK\n")
        return True
    except ImportError as e:
        print(f"FAILED: {e}")
        print("Run: pip install -r requirements.txt\n")
        return False


def check_dataset():
    print("== Checking dataset path ==")
    import config
    if not os.path.isdir(config.DATA_ROOT):
        print(f"FAILED: {config.DATA_ROOT} does not exist.")
        print("Download HC18 from https://zenodo.org/records/1327317, unzip,")
        print("and set config.DATA_ROOT to the 'training_set' folder.\n")
        return False

    from dataset import build_split_lists
    try:
        train, val, test = build_split_lists(
            config.DATA_ROOT, config.TRAIN_FRAC, config.VAL_FRAC,
            config.TEST_FRAC, config.RANDOM_SEED
        )
        print(f"Found {len(train) + len(val) + len(test)} image/mask pairs total")
        print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
        if len(train) + len(val) + len(test) < 900:
            print("WARNING: expected ~999 pairs from HC18 training_set. "
                  "Check that all files downloaded correctly.")
        print("OK\n")
        return True
    except Exception as e:
        print(f"FAILED: {e}\n")
        return False


def check_mask_filling():
    print("== Checking ellipse-outline-to-filled-mask conversion ==")
    import config
    from dataset import HC18Dataset, get_eval_transforms, build_split_lists

    train, val, test = build_split_lists(
        config.DATA_ROOT, config.TRAIN_FRAC, config.VAL_FRAC,
        config.TEST_FRAC, config.RANDOM_SEED
    )
    ds = HC18Dataset(train[:1], get_eval_transforms(config.IMAGE_SIZE))
    image, mask, meta = ds[0]
    fg_pixels = mask.sum().item()
    total_pixels = mask.numel()
    fg_ratio = fg_pixels / total_pixels
    print(f"Sample: {meta['image_path']}")
    print(f"Image tensor shape: {tuple(image.shape)}, mask shape: {tuple(mask.shape)}")
    print(f"Foreground ratio: {fg_ratio:.3f}")
    if fg_ratio < 0.02 or fg_ratio > 0.6:
        print("WARNING: unusual foreground ratio -- inspect the mask filling logic "
              "(see config.PLOT_DIR after running this for a visual check).")
    else:
        print("OK (looks like a plausible head-sized region)")

    # Save a visual for manual confirmation
    import matplotlib.pyplot as plt
    import numpy as np
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    mask_np = mask.squeeze(0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Image (normalized)")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Filled mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, "sanity_check_mask.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved visual check to {out_path} -- open it and confirm the mask "
          f"looks like a filled fetal head region.\n")
    return True


def check_model_forward_backward():
    print("== Checking model forward/backward pass ==")
    import torch
    import config
    from model import build_model
    from losses import build_loss

    device = config.DEVICE
    model = build_model().to(device)
    criterion = build_loss()

    x = torch.randn(2, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    y = torch.randint(0, 2, (2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)).float().to(device)

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(logits.shape)} (expected: (2, 1, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}))")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Total parameters: {n_params:,}")
    print("Backward pass completed without error.")
    print("OK\n")
    return True


def main():
    results = []
    results.append(("imports", check_imports()))
    if not results[-1][1]:
        sys.exit(1)
    results.append(("dataset", check_dataset()))
    if not results[-1][1]:
        sys.exit(1)
    results.append(("mask_filling", check_mask_filling()))
    results.append(("model_forward_backward", check_model_forward_backward()))

    print("== Summary ==")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if all(ok for _, ok in results):
        print("\nAll checks passed. You can now run: python train.py")
    else:
        print("\nSome checks failed -- fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
