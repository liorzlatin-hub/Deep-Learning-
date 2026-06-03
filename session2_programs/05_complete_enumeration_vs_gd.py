import numpy as np


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def complete_enumeration(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best_w, best_b, best_loss = 0.0, 0.0, float("inf")

    for w in np.linspace(-2, 6, 161):
        for b in np.linspace(-2, 4, 121):
            loss = mse(y, predict(x, w, b))
            if loss < best_loss:
                best_w, best_b, best_loss = float(w), float(b), float(loss)

    return best_w, best_b, best_loss


def gradient_descent(x: np.ndarray, y: np.ndarray, lr: float = 0.005, epochs: int = 2000) -> tuple[float, float, float]:
    w, b = 0.0, 0.0
    n = len(x)

    for _ in range(epochs):
        y_hat = predict(x, w, b)
        err = y_hat - y

        grad_w = (2.0 / n) * np.sum(err * x)
        grad_b = (2.0 / n) * np.sum(err)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mse(y, predict(x, w, b))


def main() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + rng.normal(0, 0.8, size=x.shape)

    ew, eb, eloss = complete_enumeration(x, y)
    gw, gb, gloss = gradient_descent(x, y)

    print("Complete enumeration result")
    print(f"w={ew:.3f}, b={eb:.3f}, mse={eloss:.4f}")
    print()
    print("Gradient descent result")
    print(f"w={gw:.3f}, b={gb:.3f}, mse={gloss:.4f}")


if __name__ == "__main__":
    main()
