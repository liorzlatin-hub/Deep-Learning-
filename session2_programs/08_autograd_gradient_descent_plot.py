from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_with_torch() -> tuple[list[float], float, float]:
    import torch

    torch.manual_seed(0)
    x = torch.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + 0.8 * torch.randn_like(x)

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.01
    losses: list[float] = []

    for _ in range(500):
        y_hat = w * x + b
        loss = ((y_hat - y) ** 2).mean()
        losses.append(loss.item())

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    return losses, float(w.item()), float(b.item())


def finite_diff_grad(loss_fn, w: float, b: float, eps: float = 1e-5) -> tuple[float, float]:
    dw = (loss_fn(w + eps, b) - loss_fn(w - eps, b)) / (2 * eps)
    db = (loss_fn(w, b + eps) - loss_fn(w, b - eps)) / (2 * eps)
    return dw, db


def run_fallback() -> tuple[list[float], float, float]:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 2.3 * x + 0.7 + rng.normal(0, 0.8, size=x.shape)

    def loss_fn(w: float, b: float) -> float:
        y_hat = w * x + b
        return float(np.mean((y_hat - y) ** 2))

    w, b = 0.0, 0.0
    lr = 0.01
    losses: list[float] = []

    for _ in range(500):
        dw, db = finite_diff_grad(loss_fn, w, b)
        w -= lr * dw
        b -= lr * db
        losses.append(loss_fn(w, b))

    return losses, w, b


def main() -> None:
    backend = "torch"
    try:
        losses, w, b = run_with_torch()
    except Exception as exc:
        backend = "finite-diff fallback"
        print("Torch unavailable, using fallback:", exc)
        losses, w, b = run_fallback()

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses)
    plt.title(f"Autograd Training Convergence ({backend})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(out_dir / "08_autograd_convergence.png", dpi=140)
    plt.close()

    print(f"Backend: {backend}")
    print(f"Learned parameters: w={w:.3f}, b={b:.3f}")
    print("Saved artifacts/08_autograd_convergence.png")


if __name__ == "__main__":
    main()
