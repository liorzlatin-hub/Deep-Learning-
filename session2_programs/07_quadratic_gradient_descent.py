import numpy as np


def f(x: float) -> float:
    return (x - 2.0) ** 2 + 1.0


def grad_f(x: float) -> float:
    return 2.0 * (x - 2.0)


def run_gd(x0: float, lr: float, steps: int = 20) -> tuple[float, list[float]]:
    x = x0
    trajectory = [x]
    for _ in range(steps):
        x = x - lr * grad_f(x)
        trajectory.append(x)
    return x, trajectory


def main() -> None:
    configs = [(-4.0, 0.4), (-1.0, 0.55)]

    for x0, lr in configs:
        x_final, traj = run_gd(x0, lr)
        print(f"x0={x0}, lr={lr}")
        print(f"x_final={x_final:.6f}, f(x_final)={f(x_final):.6f}")
        print("first 5 trajectory values:", [round(v, 4) for v in traj[:5]])
        print()


if __name__ == "__main__":
    main()
