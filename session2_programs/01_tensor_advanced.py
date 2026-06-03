import numpy as np


def is_contiguous(arr: np.ndarray) -> bool:
    """Simple check: C contiguous means right-most axis changes fastest in memory."""
    return arr.flags["C_CONTIGUOUS"]


def main() -> None:
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    print("Original tensor")
    print("shape:", x.shape)
    print("strides:", x.strides)
    print("is_contiguous:", is_contiguous(x))
    print()

    y = x.transpose(0, 2, 1)
    print("Transposed tensor (swap last two dimensions)")
    print("shape:", y.shape)
    print("strides:", y.strides)
    print("is_contiguous:", is_contiguous(y))
    print()

    z = np.ascontiguousarray(y)
    print("Made contiguous copy")
    print("shape:", z.shape)
    print("strides:", z.strides)
    print("is_contiguous:", is_contiguous(z))


if __name__ == "__main__":
    main()
