# import os
# import pandas as pd
# from typing import Generator, Any
# import gc

# gc.enable()

# DATA_PATHS = [
#     r'./data/monash/covid_deaths_dataset/',
#     r'./data/monash/dominick_dataset/',
#     r'./data/monash/fred_md_dataset/',
#     r'./data/monash/kaggle_web_traffic_dataset_without_missing_values/',
#     r'./data/monash/kaggle_web_traffic_weekly_dataset/',
#     r'./data/monash/kdd_cup_2018_dataset_without_missing_values/',
#     r'./data/monash/pedestrian_counts_dataset/',
#     r'./data/monash/rideshare_dataset_without_missing_values/',
#     r'./data/monash/saugeenday_dataset/',
#     r'./data/monash/solar_4_seconds_dataset/',
#     r'./data/monash/temperature_rain_dataset_without_missing_values/',
#     r'./data/monash/traffic_hourly_dataset/',
#     r'./data/monash/traffic_weekly_dataset/',
#     r'./data/monash/us_births_dataset/',
#     r'./data/monash/weather_dataset/',
#     r'./data/monash/weather_dataset/',
#     r'./data/monash/wind_4_seconds_dataset/',
#     r'./data/monash/australian_electricity_demand_dataset/',
#     r'./data/monash/vehicle_trips_dataset_without_missing_values/',
#     r'./data/monash/nn5_daily_dataset_without_missing_values/',
#     r'./data/monash/sunspot_dataset_without_missing_values/'
# ]


# def yield_df(directory: str) -> Generator[pd.DataFrame, Any, None]:
#     """
#     Generator function that yields individual CSV files from a directory as pandas DataFrames.

#     Args:
#         directory (str): Path to the directory containing CSV files

#     Yields:
#         pandas.DataFrame: DataFrame containing the contents of each CSV file
#     """
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.csv'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     yield pd.read_csv(file_path)
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")
#                 finally:
#                     gc.collect()


# for directory in DATA_PATHS:
#     print(f"Processing directory: {directory}")
#     for df in yield_df(directory):
#         try:
#             print(f"Processing file in {directory}, Shape: {df.shape}")
#             pass
#         except Exception as e:
#             print(f"Error processing DataFrame from {directory}: {e}")
#             pass
#         finally:
#             del df 
#             gc.collect() 
#     del directory
#     gc.collect()

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from optimization_methods import (
    random_search, bayesian_optimization_gp, bohb_kde, tpe_optimization,
    cma_es_optimization, differential_evolution, hyperband, async_hyperband,
    dehb_optimization, successive_halving, dragonfly_optimization, multi_fidelity_tpe,
    pso_optimization
)

# Generate synthetic time series data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
X = np.column_stack([
    t, np.sin(t), np.cos(t), np.random.normal(0, 0.1, n_samples)
])
y = 2 * np.sin(t) + np.cos(t) + np.random.normal(0, 0.2, n_samples)

# Evaluation function using TimeSeriesSplit
def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred, squared=False))
    return np.mean(scores)

# Define all optimization methods to test
optimization_methods = [
    ('Random Search', random_search),
    ('Bayesian Optimization (GP)', bayesian_optimization_gp),
    ('BOHB (KDE)', bohb_kde),
    ('TPE', tpe_optimization),
    ('CMA-ES', cma_es_optimization),
    ('Differential Evolution', differential_evolution),
    ('Hyperband', hyperband),
    ('Async Hyperband', async_hyperband),
    ('DEHB', dehb_optimization),
    ('Successive Halving', successive_halving),
    ('Dragonfly (BO)', dragonfly_optimization),
    ('Multi-Fidelity TPE', multi_fidelity_tpe),
    ('Particle Swarm Optimization', pso_optimization)
]

# Initialize results storage
results = []

# Run tests for each method
model = RandomForestRegressor(random_state=42)
n_iter = 10
n_splits = 5

for method_name, method_func in optimization_methods:
    print(f"Testing {method_name}...")
    start_time = time.time()
    try:
        best_model, best_params, best_score = method_func(model, X, y, n_iter=n_iter, n_splits=n_splits)
        runtime = time.time() - start_time
        results.append({
            'Method': method_name,
            'Best Parameters': best_params,
            'RMSE': best_score,
            'Runtime (s)': runtime
        })
        print(f"{method_name} completed. RMSE: {best_score:.4f}, Runtime: {runtime:.2f}s")
    except Exception as e:
        print(f"{method_name} failed: {str(e)}")
        results.append({
            'Method': method_name,
            'Best Parameters': None,
            'RMSE': None,
            'Runtime (s)': time.time() - start_time
        })

# Create summary table
results_df = pd.DataFrame(results)
results_df = results_df[['Method', 'RMSE', 'Runtime (s)', 'Best Parameters']]
print("\nSummary of Results:")
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('hyperparameter_optimization_results.csv', index=False)

# Find best method
if results_df['RMSE'].notnull().any():
    best_method = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"\nBest Method: {best_method['Method']}")
    print(f"Best RMSE: {best_method['RMSE']:.4f}")
    print(f"Best Parameters: {best_method['Best Parameters']}")