from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray) -> np.ndarray:
    return (x - 2.0) ** 2 + 1.0


def grad_f(x: float) -> float:
    return 2.0 * (x - 2.0)


def run_gd(x0: float, lr: float, steps: int = 20) -> np.ndarray:
    x = x0
    trajectory = [x]
    for _ in range(steps):
        x = x - lr * grad_f(x)
        trajectory.append(x)
    return np.array(trajectory)


def main() -> None:
    configs = [(-4.0, 0.4), (-1.0, 0.55)]
    x_curve = np.linspace(-6, 6, 400)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_curve, f(x_curve), label="f(x)=(x-2)^2+1")

    for x0, lr in configs:
        traj = run_gd(x0, lr)
        plt.scatter(traj, f(traj), s=18, label=f"x0={x0}, lr={lr}")

    plt.title("Quadratic Gradient Descent Trajectories")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "07_quadratic_trajectories.png", dpi=140)
    plt.close()

    print("Saved artifacts/07_quadratic_trajectories.png")


if __name__ == "__main__":
    main()
