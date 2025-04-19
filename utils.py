import pickle
import pandas as pd
from typing import Generator, Any
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from PyEMD import EMD
from pmdarima import auto_arima
import tsfel
from tsfresh import extract_features
import pywt
from scipy import signal
import warnings
from os import cpu_count
warnings.filterwarnings('ignore')

CPU_COUNT = cpu_count()


def yield_data(
        pickle_file_path="./data/monash/monash-df.pkl") -> Generator[dict[str, Any], Any, None]:
    """
    ## Generator function to yield objects one at a time from a pickle file.

    ### Args:
        - pickle_file_path (`str`): Path to the pickle file.

    ### Yields:
        - `dict`: A dictionary with file name, df, freq as keys.

    ### Raises:
        - `FileNotFoundError`: If the pickle file doesn't exist.
        - `pickle.UnpicklingError`: If the pickle file is corrupted.
    """
    try:
        with open(pickle_file_path, "rb") as f:
            while True:
                try:
                    items = list(pickle.load(f).items())[0]
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
        raise pickle.UnpicklingError("Corrupted pickle file. Re-run setup.sh")


def prepare_time_series(df, frequency) -> Generator[pd.DataFrame, Any, None]:
    """
    ## Generator function to yield individual time series DataFrames from a DataFrame with series_value lists.

    ### Args:
        - df (`pd.DataFrame`): DataFrame with series_name, start_timestamp, series_value.
        - frequency (`str`): Frequency of the series (e.g., '4_seconds', 'half_hourly', 'daily').

    ### Yields:
        - `pd.DataFrame`: A DataFrame with timestamps as index and a single column for the series.
    """
    freq_map = {
        '4_seconds': '4s',
        'minutely': 'min',
        'hourly': 'h',
        'half_hourly': '30min',
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',
        'quarterly': 'QE',
        'yearly': 'YE'
    }

    pandas_freq = freq_map[frequency]
    for _, row in df.iterrows():
        start_time = pd.Timestamp.min
        values = row['series_value']
        try:
            yield pd.DataFrame(
                {row['series_name']: pd.Series(
                    values, index=pd.date_range(
                        start=start_time, periods=len(values), freq=pandas_freq
                    )
                )
                }
            )
        except (pd.errors.OutOfBoundsDatetime, OverflowError):
            try:
                lo, hi = 1, len(values)
                max_len = 0

                while lo <= hi:
                    mid = (lo + hi) // 2
                    try:
                        pd.date_range(start=start_time,
                                      periods=mid, freq=pandas_freq)
                        max_len = mid
                        lo = mid + 1
                    except (pd.errors.OutOfBoundsDatetime, OverflowError):
                        hi = mid - 1

                if max_len == 0:
                    continue
                yield pd.DataFrame(
                    {row['series_name']: pd.Series(
                        values[:max_len], index=pd.date_range(
                            start=start_time, periods=max_len, freq=pandas_freq
                        )
                    )
                    }
                )
            except Exception as inner_e:
                print(
                    f"Failed to truncate and recover series {row['series_name']}: {str(inner_e)}")
                continue
        except Exception as e:
            print(e)
            continue


def auto_featurize(df) -> pd.DataFrame:
    """
    Automatically featurize a time series DataFrame with:
    - One column containing timestamps (automatically detected)
    - One column containing values (automatically detected)
    """

    value_col = df.columns[0]

    # 1. Basic Datetime Features
    df['second'] = df.index.second
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week
    df['half_hourly'] = (df.index.hour * 2) + (df.index.minute // 30)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # 2. Lag Features
    lags = [1, 2, 3, 24, 24 * 7]
    for lag in lags:
        df[f'lag_{lag}'] = df[value_col].shift(lag)

    # 3. Rolling Window Statistics
    windows = [3, 7, 24, 24 * 7]
    for window in windows:
        df[f'rolling_mean_{window}'] = df[value_col].rolling(window).mean()
        df[f'rolling_std_{window}'] = df[value_col].rolling(window).std()
        df[f'rolling_min_{window}'] = df[value_col].rolling(window).min()
        df[f'rolling_max_{window}'] = df[value_col].rolling(window).max()

    # 4. Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(
            df[value_col].dropna(),
            model='additive',
            period=24)
        df['trend'] = decomposition.trend
        df['seasonal'] = decomposition.seasonal
        df['residual'] = decomposition.resid
    except Exception as e:
        print(f"Seasonal decomposition failed: {e}")

    # 5. Empirical Mode Decomposition (EMD)
    try:
        emd = EMD()
        imfs = emd(df[value_col].dropna().values)
        for i in range(imfs.shape[0]):
            df[f'imf_{i+1}'] = np.nan
            valid_length = min(len(imfs[i]), len(df))
            df.iloc[-valid_length:,
                    df.columns.get_loc(f'imf_{i+1}')] = imfs[i][-valid_length:]
    except Exception as e:
        print(f"EMD failed: {e}")

    # 6. ARIMA Residuals
    try:
        model = auto_arima(df[value_col].dropna(),
                           seasonal=False,
                           suppress_warnings=True,
                           n_jobs=-1)
        df['arima_residual'] = np.nan
        valid_length = min(len(model.resid()), len(df))
        df.iloc[-valid_length:,
                df.columns.get_loc('arima_residual')] = model.resid()[-valid_length:]
    except Exception as e:
        print(f"ARIMA failed: {e}")

    # # 7. TSFresh Features
    # try:
    #     tsfresh_df = df[[value_col]].reset_index()
    #     tsfresh_df['id'] = 1  # Single time series
    #     features = extract_features(tsfresh_df,
    #                                 column_id='id',
    #                                 column_sort='index',  # Fixed from 'timestamp'
    #                                 column_value=value_col,)
    #     feature_values = features.iloc[0].to_dict()
    #     for key, value in feature_values.items():
    #         df[f'tsfresh_{key}'] = value  # Add as constant columns
    # except Exception as e:
    #     print(f"TSFresh failed: {e}")

    # 8. TSFEL Features
    try:
        cfg = tsfel.get_features_by_domain()
        window_size = 24 * 7  # Example window size
        tsfel_features = tsfel.time_series_features_extractor(
            cfg, df[value_col], window_size=window_size, overlap=0.5, n_jobs=CPU_COUNT, )
        # Removed incorrect index assignment
        df = df.join(tsfel_features.add_prefix('tsfel_'))
    except Exception as e:
        print(f"TSFEL failed: {e}")

    # 9. FFT Features
    try:
        window_size = 24  # Example window
        fft_features = []
        for i in range(len(df) - window_size + 1):
            window = df[value_col].iloc[i:i + window_size]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            fft_features.append({
                'fft_peak_freq': np.argmax(amplitudes[1:window_size // 2]),
                'fft_peak_amp': np.max(amplitudes[1:window_size // 2])
            })
        fft_df = pd.DataFrame(fft_features, index=df.index[window_size - 1:])
        df = df.join(fft_df)
    except Exception as e:
        print(f"FFT failed: {e}")

    # 10. Wavelet Transform
    try:
        coeffs, _ = pywt.cwt(
            df[value_col].dropna().values, np.arange(
                1, 10), 'gaus1')
        for i in range(coeffs.shape[0]):
            df[f'wavelet_{i}'] = np.nan
            valid_length = min(len(coeffs[i]), len(df))
            df.iloc[-valid_length:,
                    df.columns.get_loc(f'wavelet_{i}')] = coeffs[i][-valid_length:]
    except Exception as e:
        print(f"Wavelet failed: {e}")

    # 11. Microstructure-FFT Fusion
    try:
        micro_window = 5  # Small window for microstructure
        micro_features = []
        for i in range(len(df) - micro_window + 1):
            window = df[value_col].iloc[i:i + micro_window]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            micro_features.append({
                'micro_fft_peak': np.argmax(amplitudes[1:micro_window // 2]),
                'micro_fft_amp': np.max(amplitudes[1:micro_window // 2])
            })
        micro_df = pd.DataFrame(micro_features,
                                index=df.index[micro_window - 1:])
        df = df.join(micro_df)
    except Exception as e:
        print(f"Microstructure FFT failed: {e}")

    # Final cleaning
    df = df.ffill().bfill().dropna(how='all', axis=1)

    return df
