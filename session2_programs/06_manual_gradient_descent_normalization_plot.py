from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def fit_linear_gd(x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> tuple[float, float, list[float]]:
    w, b = 0.0, 0.0
    n = len(x)
    history: list[float] = []

    for _ in range(epochs):
        y_hat = w * x + b
        err = y_hat - y

        loss = float(np.mean(err**2))
        history.append(loss)

        grad_w = (2.0 / n) * np.sum(err * x)
        grad_b = (2.0 / n) * np.sum(err)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, history


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def main() -> None:
    rng = np.random.default_rng(12)
    x_raw = np.linspace(1000, 5000, 120)
    y = 0.04 * x_raw + 5.0 + rng.normal(0, 3.0, size=x_raw.shape)

    _, _, raw_hist = fit_linear_gd(x_raw, y, lr=1e-8, epochs=500)

    x_norm = zscore(x_raw)
    _, _, norm_hist = fit_linear_gd(x_norm, y, lr=0.03, epochs=500)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(raw_hist, label="Raw input")
    plt.plot(norm_hist, label="Normalized input")
    plt.yscale("log")
    plt.title("Normalization Effect on Gradient Descent")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "06_normalization_convergence.png", dpi=140)
    plt.close()

    print("Saved artifacts/06_normalization_convergence.png")
    print("Final loss raw:", round(raw_hist[-1], 4))
    print("Final loss norm:", round(norm_hist[-1], 4))


if __name__ == "__main__":
    main()
