from joblib import Parallel, delayed
from utils import yield_data, prepare_time_series


def compute_nan_percent(series_df) -> float:
    """Compute % NaNs in a single univariate time-series DataFrame."""
    n_missing = series_df.isna().sum().sum()
    n_total = series_df.size
    return (n_missing / n_total) * 100 if n_total > 0 else 0


nan_percentages = {}
i = 0
for data in yield_data():
    series_list = list(prepare_time_series(data['df'], data['freq']))
    series_count = len(series_list)

    if series_count == 0:
        nan_percentages[i] = 0
        print(f"Dataset {i} — No series found.")
        i += 1
        continue

    nan_percents = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_nan_percent)(series) for series in series_list
    )

    avg_nan_percent = sum(nan_percents) / series_count
    nan_percentages[i] = avg_nan_percent
    print(f"Dataset {i} — Avg % NaNs: {avg_nan_percent:.2f}")
    i += 1

print("\nSummary of Average % NaNs per Dataset:")
for key, val in sorted(nan_percentages.items()):
    print(f"Dataset {key}: {val:.2f}%")
