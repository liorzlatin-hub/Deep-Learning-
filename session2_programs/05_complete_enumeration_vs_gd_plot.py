from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def gradient_descent_with_history(
    x: np.ndarray, y: np.ndarray, lr: float = 0.005, epochs: int = 2000
) -> tuple[float, float, list[float]]:
    w, b = 0.0, 0.0
    n = len(x)
    losses: list[float] = []

    for _ in range(epochs):
        y_hat = predict(x, w, b)
        err = y_hat - y
        losses.append(float(np.mean(err**2)))

        grad_w = (2.0 / n) * np.sum(err * x)
        grad_b = (2.0 / n) * np.sum(err)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, losses


def main() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + rng.normal(0, 0.8, size=x.shape)

    w, b, losses = gradient_descent_with_history(x, y)
    y_hat = predict(x, w, b)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses)
    plt.title("Gradient Descent Convergence (Linear Regression)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(out_dir / "05_convergence.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.scatter(x, y, s=18, alpha=0.7, label="Data")
    plt.plot(x, y_hat, color="red", linewidth=2, label="Fitted line")
    plt.title("Complete Enumeration Topic - GD Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "05_fit.png", dpi=140)
    plt.close()

    print(f"Learned parameters: w={w:.3f}, b={b:.3f}")
    print("Saved artifacts/05_convergence.png and artifacts/05_fit.png")


if __name__ == "__main__":
    main()
