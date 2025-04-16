import os
import multiprocessing as mp
from utils import yield_data, prepare_time_series


def compute_nan_percent(series_df):
    """Compute % NaNs in a single univariate time-series DataFrame."""
    n_missing = series_df.isna().sum().sum()
    n_total = series_df.size
    return (n_missing / n_total) * 100 if n_total > 0 else 0


nan_percentages = {}
cpu_count = os.cpu_count()
i = 0
for data in yield_data():
    series_list = list(prepare_time_series(data['df'], data['freq']))
    series_count = len(series_list)

    if series_count == 0:
        nan_percentages[i] = 0
        print(f"Dataset {i} — No series found.")
        i += 1
        continue

    with mp.Pool(processes=cpu_count) as pool:
        nan_percents = pool.map(compute_nan_percent, series_list)

    avg_nan_percent = sum(nan_percents) / series_count
    nan_percentages[i] = avg_nan_percent
    print(f"Dataset {i} — Avg % NaNs: {avg_nan_percent:.2f}")
    i += 1

print("\nSummary of Average % NaNs per Dataset:")
for key, val in sorted(nan_percentages.items()):
    print(f"Dataset {key}: {val:.2f}%")
