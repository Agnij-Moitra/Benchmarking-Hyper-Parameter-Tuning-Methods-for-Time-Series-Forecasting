import pickle
import pandas as pd


def yield_data(pickle_file_path="./data/monash/monash-df.pkl"):
    """
    ## Generator function to yield objects one at a time from a pickle file.

    ### Args:
        - pickle_file_path (str): Path to the pickle file.

    ### Yields:
        - dict: A dictionary with file name, df, freq as keys.

    ### Raises:
        - FileNotFoundError: If the pickle file doesn't exist.
        - pickle.UnpicklingError: If the pickle file is corrupted.
    """
    try:
        with open(pickle_file_path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    items = [i for i in obj.items()][0]
                    yield {
                        "name": items[0].split('.')[0],
                        "df": items[1][0],
                        "freq": items[1][1],
                    }
                except EOFError:
                    break
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {pickle_file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError("Corrupted pickle file.")


def prepare_time_series(df, frequency):
    """
    ## Generator function to yield individual time series DataFrames from a DataFrame with series_value lists.

    ### Args:
        - df (pd.DataFrame): DataFrame with series_name, start_timestamp, series_value.
        - frequency (str): Frequency of the series (e.g., '4_seconds', 'half_hourly', 'daily').

    ### Yields:
        - pd.DataFrame: A DataFrame with timestamps as index and a single column for the series.

    ### Raises:
        - ValueError: If the frequency is unsupported.
    """
    freq_map = {
        '4_seconds': '4s',
        'minutely': 'min',
        'hourly': 'h',
        'half_hourly': '30min',
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',
        'quarterly': 'Q',
        'yearly': 'Y'
    }

    pandas_freq = freq_map.get(frequency)
    if pandas_freq is None:
        raise ValueError(f"Unsupported frequency: {frequency}")

    for _, row in df.iterrows():
        series_name = row['series_name']
        try:
            start_time = pd.to_datetime(row['start_timestamp'])
        except KeyError:
            start_time = pd.Timestamp("2000-01-01 00:00:00")

        values = row['series_value']
        return pd.DataFrame(
            {series_name: pd.Series(
                values, index=pd.date_range(
                    start=start_time, periods=len(values), freq=pandas_freq
                    ))
            }
            )
