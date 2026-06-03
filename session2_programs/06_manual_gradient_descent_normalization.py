import numpy as np


def fit_linear_gd(x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> tuple[float, float, list[float]]:
    w, b = 0.0, 0.0
    n = len(x)
    history = []

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

    print("Final loss without normalization:", round(raw_hist[-1], 4))
    print("Final loss with normalization:", round(norm_hist[-1], 4))
    print("Loss at epoch 1 (raw vs norm):", round(raw_hist[0], 4), "vs", round(norm_hist[0], 4))


if __name__ == "__main__":
    main()
