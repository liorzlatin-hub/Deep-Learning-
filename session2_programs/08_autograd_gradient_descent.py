import numpy as np


def run_with_torch() -> None:
    import torch

    torch.manual_seed(0)

    x = torch.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + 0.8 * torch.randn_like(x)

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.01

    for epoch in range(500):
        y_hat = w * x + b
        loss = ((y_hat - y) ** 2).mean()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 100 == 0:
            print(f"epoch={epoch + 1}, loss={loss.item():.4f}")

    print(f"Learned parameters: w={w.item():.3f}, b={b.item():.3f}")


def finite_diff_grad(loss_fn, w: float, b: float, eps: float = 1e-5) -> tuple[float, float]:
    dw = (loss_fn(w + eps, b) - loss_fn(w - eps, b)) / (2 * eps)
    db = (loss_fn(w, b + eps) - loss_fn(w, b - eps)) / (2 * eps)
    return dw, db


def run_fallback() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + rng.normal(0, 0.8, size=x.shape)

    def loss_fn(w: float, b: float) -> float:
        y_hat = w * x + b
        return float(np.mean((y_hat - y) ** 2))

    w, b = 0.0, 0.0
    lr = 0.01

    for epoch in range(500):
        dw, db = finite_diff_grad(loss_fn, w, b)
        w -= lr * dw
        b -= lr * db

        if (epoch + 1) % 100 == 0:
            print(f"epoch={epoch + 1}, loss={loss_fn(w, b):.4f}")

    print(f"Approx learned parameters: w={w:.3f}, b={b:.3f}")


def main() -> None:
    try:
        run_with_torch()
    except Exception as exc:
        print("Torch path unavailable, running finite-difference fallback.")
        print("Reason:", exc)
        run_fallback()


if __name__ == "__main__":
    main()
