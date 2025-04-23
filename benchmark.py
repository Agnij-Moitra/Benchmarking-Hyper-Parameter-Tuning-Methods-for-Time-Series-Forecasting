import os
import pandas as pd
from typing import Generator, Any
import gc

gc.enable()

DATA_PATHS = [
    r'./data/monash/covid_deaths_dataset/',
    r'./data/monash/dominick_dataset/',
    r'./data/monash/fred_md_dataset/',
    r'./data/monash/kaggle_web_traffic_dataset_without_missing_values/',
    r'./data/monash/kaggle_web_traffic_weekly_dataset/',
    r'./data/monash/kdd_cup_2018_dataset_without_missing_values/',
    r'./data/monash/pedestrian_counts_dataset/',
    r'./data/monash/rideshare_dataset_without_missing_values/',
    r'./data/monash/saugeenday_dataset/',
    r'./data/monash/solar_4_seconds_dataset/',
    r'./data/monash/solar_weekly_dataset/',
    r'./data/monash/temperature_rain_dataset_without_missing_values/',
    r'./data/monash/traffic_hourly_dataset/',
    r'./data/monash/traffic_weekly_dataset/',
    r'./data/monash/us_births_dataset/',
    r'./data/monash/weather_dataset/',
    r'./data/monash/weather_dataset/',
    r'./data/monash/wind_4_seconds_dataset/',
    r'./data/monash/australian_electricity_demand_dataset/',
    r'./data/monash/vehicle_trips_dataset_without_missing_values/',
    r'./data/monash/nn5_daily_dataset_without_missing_values/',
    r'./data/monash/sunspot_dataset_without_missing_values/'
]


def yield_df(directory: str) -> Generator[pd.DataFrame, Any, None]:
    """
    Generator function that yields individual CSV files from a directory as pandas DataFrames.

    Args:
        directory (str): Path to the directory containing CSV files

    Yields:
        pandas.DataFrame: DataFrame containing the contents of each CSV file
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    yield pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                finally:
                    gc.collect()


for directory in DATA_PATHS:
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist, skipping...")
        continue

    print(f"Processing directory: {directory}")
    # df_list = list(yield_df(directory))
    for df in yield_df(directory):
        try:
            # Example processing: Print the shape and first few rows of each DataFrame
            print(f"Processing file in {directory}, Shape: {df.shape}")
            pass
        except Exception as e:
            print(f"Error processing DataFrame from {directory}: {e}")
            pass
        finally:
            del df  # Explicitly delete DataFrame to free memory
            gc.collect()  # Run garbage collection to ensure memory is freed
    del directory
    gc.collect()
