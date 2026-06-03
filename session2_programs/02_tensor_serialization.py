from pathlib import Path
import numpy as np


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    features = np.random.default_rng(42).normal(size=(5, 3)).astype(np.float32)
    labels = np.array([0, 1, 1, 0, 1], dtype=np.int64)

    file_path = out_dir / "sample_dataset.npz"
    np.savez(file_path, features=features, labels=labels, description="session2 serialization demo")

    loaded = np.load(file_path)
    loaded_features = loaded["features"]
    loaded_labels = loaded["labels"]
    description = loaded["description"]

    print("Saved:", file_path)
    print("features shape:", loaded_features.shape)
    print("labels shape:", loaded_labels.shape)
    print("description:", description)


if __name__ == "__main__":
    main()
