from pathlib import Path
import numpy as np


def to_grayscale(chw: np.ndarray) -> np.ndarray:
    """Convert [C,H,W] RGB to grayscale [H,W] using luminance weights."""
    r, g, b = chw[0], chw[1], chw[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def main() -> None:
    rng = np.random.default_rng(0)

    # Synthetic image tensor in [C,H,W] layout.
    image = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)

    image_float = image.astype(np.float32) / 255.0
    gray = to_grayscale(image_float)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "image_tensor_chw.npy", image)
    np.save(out_dir / "image_gray.npy", gray)

    print("image dtype:", image.dtype)
    print("image shape [C,H,W]:", image.shape)
    print("normalized range:", (float(image_float.min()), float(image_float.max())))
    print("gray shape [H,W]:", gray.shape)
    print("Saved artifacts/image_tensor_chw.npy and artifacts/image_gray.npy")


if __name__ == "__main__":
    main()
