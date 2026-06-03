import numpy as np


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(len(series) - window):
        xs.append(series[i : i + window])
        ys.append(series[i + window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    rng = np.random.default_rng(123)
    t = np.arange(0, 200)

    # Simple synthetic time series: trend + seasonality + noise.
    series = 0.02 * t + np.sin(0.15 * t) + 0.1 * rng.normal(size=t.shape)

    window = 12
    x, y = make_windows(series, window)

    print("series length:", len(series))
    print("window size:", window)
    print("X shape [num_samples, window]:", x.shape)
    print("y shape [num_samples]:", y.shape)
    print("first X sample:", np.round(x[0], 3))
    print("first y sample:", round(float(y[0]), 3))


if __name__ == "__main__":
    main()
