from time import perf_counter
import pickle
import pandas as pd
from typing import Generator, Any
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from PyEMD import EMD
from pmdarima import auto_arima
import tsfel
import pywt
from scipy import signal
from joblib import Parallel, delayed
import os
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import pacf, acf
import warnings
from os import cpu_count
import random
random.seed(1027)
np.random.seed(1027)
warnings.filterwarnings('ignore')

CPU_COUNT = cpu_count()

# Original FREQ_CONFIG and FREQ_MAP
FREQ_CONFIG = {
    '4_seconds': {
        'lags': [1, 15, 30, 60 * 15],  # 4s, 1m, 2m, 1h
        'rolling_windows': [15, 30, 60 * 15],
        'seasonal_period': 60 * 15 * 24,  # 24h
        'fft_window': 60 * 15,  # 1h
        'micro_window': 5,  # 20s
        'tsfel_window': 60 * 15 * 24  # 24h
    },
    'minutely': {
        'lags': [1, 60, 60 * 24, 60 * 24 * 7],  # 1m, 1h, 1d, 1w
        'rolling_windows': [60, 60 * 24, 60 * 24 * 7],
        'seasonal_period': 60 * 24,  # Daily
        'fft_window': 60 * 24,  # 1d
        'micro_window': 15,  # 15m
        'tsfel_window': 60 * 24 * 7  # 1w
    },
    'hourly': {
        'lags': [1, 24, 24 * 7],  # 1h, 1d, 1w
        'rolling_windows': [3, 24, 24 * 7],
        'seasonal_period': 24,  # Daily
        'fft_window': 24 * 7,  # 1w
        'micro_window': 6,  # 6h
        'tsfel_window': 24 * 7  # 1w
    },
    'half_hourly': {
        'lags': [1, 48, 48 * 7],  # 30min, 1d, 1w
        'rolling_windows': [48, 48 * 7],
        'seasonal_period': 48,  # Daily
        'fft_window': 48 * 7,  # 1w
        'micro_window': 12,  # 6h
        'tsfel_window': 48 * 7  # 1w
    },
    'daily': {
        'lags': [1, 7, 30],  # 1d, 1w, ~1m
        'rolling_windows': [7, 30, 30 * 4, 30 * 12],
        'seasonal_period': 7,  # Weekly
        'fft_window': 365,  # 1y
        'micro_window': 5,  # 5d
        'tsfel_window': 30  # ~1m
    },
    'weekly': {
        'lags': [1, 4, 52],  # 1w, 1m, 1y
        'rolling_windows': [4, 52, 52 * 5],
        'seasonal_period': 52,  # Yearly
        'fft_window': 52,  # 1y
        'micro_window': 4,  # 4w
        'tsfel_window': 52  # 1y
    },
    'monthly': {
        'lags': [1, 12],  # 1m, 1y
        'rolling_windows': [12, 12 * 5, 12 * 10],
        'seasonal_period': 12,  # Yearly
        'fft_window': 12,  # 1y
        'micro_window': 3,  # 3m
        'tsfel_window': 12  # 1y
    },
    'quarterly': {
        'lags': [1, 4, 4 * 2, 4 * 5],  # 1q, 1y
        'rolling_windows': [4, 4 * 3, 4 * 3 * 5],
        'seasonal_period': 4,  # Yearly
        'fft_window': 4,  # 1y
        'micro_window': 4,  # 1y
        'tsfel_window': 4  # 1y
    },
    'yearly': {
        'lags': [1, 5, 10],  # 1y, 5y
        'rolling_windows': [2, 5, 10],
        'seasonal_period': 5,  # 5y cycle
        'fft_window': 5,  # 5y
        'micro_window': 3,  # 3y
        'tsfel_window': 5  # 5y
    }
}

FREQ_MAP = {
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

WAVELETS = [
    'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8',
    'mexh',
    'morl',
    'cmor',
    'shan',
    'cmor1.5-1.0',
    'shan1.5-1.0',
    'fbsp1-1.5-1.0',
    'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8',
]


def yield_data(
        pickle_file_path="./data/monash/monash-df.pkl") -> Generator[dict[str, Any], Any, None]:
    """
    Generator function to yield objects one at a time from a pickle file.

    Args:
        pickle_file_path (str): Path to the pickle file.

    Yields:
        dict: A dictionary with file name, df, freq as keys.

    Raises:
        FileNotFoundError: If the pickle file doesn't exist.
        pickle.UnpicklingError: If the pickle file is corrupted.
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
    Generator function to yield individual time series DataFrames from a DataFrame with series_value lists.

    Args:
        df (pd.DataFrame): DataFrame with series_name, start_timestamp, series_value.
        frequency (str): Frequency of the series (e.g., '4_seconds', 'half_hourly', 'daily').

    Yields:
        pd.DataFrame: A DataFrame with timestamps as index and a single column for the series.
    """
    pandas_freq = FREQ_MAP[frequency]
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


def infer_time_series_params(
        series: pd.Series,
        frequency: str,
        min_length: int = 10) -> dict:
    """
    Infer optimal parameters for time series analysis and forecasting, ensuring parameters
    do not exceed series length constraints (len(series) > 100 * parameter).

    Args:
        series (pd.Series): Input time series with timestamps as index.
        frequency (str): Data frequency (e.g., 'daily', 'hourly', '4_seconds').
        min_length (int): Minimum series length for parameter inference.

    Returns:
        dict: Inferred parameters including lags, rolling_windows, seasonal_period,
              fft_window, micro_window, and tsfel_window.
    """
    if frequency not in FREQ_CONFIG:
        raise ValueError(
            f"Invalid frequency. Choose from {list(FREQ_CONFIG.keys())}")

    if len(series) < min_length:
        print(
            f"Series too short ({len(series)} < {min_length}). Using defaults for {frequency}.")
        return FREQ_CONFIG[frequency]

    series = series.dropna()
    N = len(series)
    min_cycles = 100

    # 1. Lag Selection Using PACF
    try:
        nlags = min(N // 2, 40)
        pacf_values = pacf(series, nlags=nlags)
        significant_lags = [i + 1 for i,
                            val in enumerate(pacf_values[1:],
                                             1) if abs(val) > 0.05]
        lags = [lag for lag in significant_lags if N > min_cycles * lag]
        if not lags:
            lags = [lag for lag in FREQ_CONFIG[frequency]
                    ['lags'] if N > min_cycles * lag]
            if not lags:
                lags = [1]  # Minimum fallback
    except Exception as e:
        print(f"PACF failed: {e}. Using default lags for {frequency}.")
        lags = [lag for lag in FREQ_CONFIG[frequency]
                ['lags'] if N > min_cycles * lag] or [1]

    # 2. Seasonal Period Detection
    try:
        # Autocorrelation analysis
        acf_values = acf(series, nlags=min(N // min_cycles, 52))
        peaks, properties = find_peaks(acf_values[1:], prominence=0.05)
        acf_periods = [p + 1 for p in peaks if N >
                       min_cycles * (p + 1)]
        acf_prominences = properties['prominences'] if acf_periods else []

        # Spectral analysis
        detrended = signal.detrend(series.values)
        freqs, spectrum = signal.periodogram(detrended)
        valid_freq_indices = np.where(
            (freqs > 0) & (freqs >= min_cycles / N))[0]
        spectral_periods = [int(np.round(1.0 / freqs[i])) for i in valid_freq_indices
                            if freqs[i] > 0 and N > min_cycles * int(np.round(1.0 / freqs[i]))]
        spectral_amplitudes = [spectrum[i]
                               for i in valid_freq_indices] if spectral_periods else []

        # Combine and select best period
        if acf_periods and spectral_periods:
            # Prioritize period with highest score (prominence for ACF,
            # amplitude for spectral)
            all_periods = acf_periods + spectral_periods
            scores = ([p for p in acf_prominences] +
                      [a / max(spectral_amplitudes) if spectral_amplitudes else 0
                      for a in spectral_amplitudes])
            valid_periods = [p for p in all_periods if N > min_cycles * p]
            if valid_periods:
                best_idx = np.argmax([scores[all_periods.index(p)]
                                     for p in valid_periods])
                seasonal_period = valid_periods[best_idx]
            else:
                seasonal_period = FREQ_CONFIG[frequency]['seasonal_period']
        elif acf_periods:
            seasonal_period = acf_periods[np.argmax(acf_prominences)]
        elif spectral_periods:
            seasonal_period = spectral_periods[np.argmax(spectral_amplitudes)]
        else:
            seasonal_period = FREQ_CONFIG[frequency]['seasonal_period']

        if seasonal_period > N // min_cycles:
            seasonal_period = FREQ_CONFIG[frequency]['seasonal_period']
    except Exception as e:
        print(
            f"Seasonal period detection failed: {e}. Using default for {frequency}.")
        seasonal_period = FREQ_CONFIG[frequency]['seasonal_period']

    # 3. Rolling Window Parameters
    try:
        base_windows = [2, 3, 4, 5, 10, 15, 30, 48, 52, 60 * 15, 60 * 24]
        rolling_windows = [w for w in base_windows if N > min_cycles * w]
        if not rolling_windows:
            rolling_windows = [min(base_windows)]  # Smallest as fallback
    except Exception as e:
        print(
            f"Rolling window inference failed: {e}. Using default for {frequency}.")
        rolling_windows = [w for w in FREQ_CONFIG[frequency]['rolling_windows']
                           if N > min_cycles * w] or [min(FREQ_CONFIG[frequency]['rolling_windows'])]

    # 4. FFT Window Selection
    try:
        possible_fft_windows = [2 ** i for i in range(int(np.log2(N)) + 1)]
        fft_windows = [w for w in possible_fft_windows if N > min_cycles * w]
        fft_window = max(fft_windows) if fft_windows else min(
            possible_fft_windows)
    except Exception as e:
        print(
            f"FFT window inference failed: {e}. Using default for {frequency}.")
        fft_window = FREQ_CONFIG[frequency]['fft_window']

    # 5. Micro Window (Short-Term Patterns)
    try:
        possible_micro_windows = [max(5, seasonal_period //
                                      10), max(5, seasonal_period //
                                               5)] if seasonal_period > 0 else [5]
        micro_windows = [
            w for w in possible_micro_windows if N > min_cycles * w]
        micro_window = min(
            micro_windows) if micro_windows else FREQ_CONFIG[frequency]['micro_window']
    except Exception:
        micro_window = FREQ_CONFIG[frequency]['micro_window']

    # 6. TSFEL Window (Feature Extraction)
    try:
        possible_tsfel_windows = [seasonal_period *
                                  2, seasonal_period *
                                  4] if seasonal_period > 0 else [max(10, N //
                                                                      5)]
        tsfel_windows = [
            w for w in possible_tsfel_windows if N > min_cycles * w]
        tsfel_window = min(
            tsfel_windows) if tsfel_windows else FREQ_CONFIG[frequency]['tsfel_window']
    except Exception:
        tsfel_window = FREQ_CONFIG[frequency]['tsfel_window']
    return {
        'lags': lags,
        'rolling_windows': rolling_windows,
        'seasonal_period': seasonal_period,
        'fft_window': fft_window,
        'micro_window': micro_window,
        'tsfel_window': tsfel_window
    }


# ====================
# Feature Extractors
# ====================

def extract_datetime_features(
        series: pd.Series,
        config: dict,
        frequency) -> pd.DataFrame:
    """Extract datetime-based features from series index."""
    features = pd.DataFrame(index=series.index)
    dt = series.index

    datetime_features = {
        'second': dt.second,
        'minute': dt.minute,
        'hour': dt.hour,
        'day_of_week': dt.dayofweek,
        'week_of_year': dt.isocalendar().week.astype(int),
        'half_hourly': (dt.hour * 2) + (dt.minute // 30),
        'month': dt.month,
        'quarter': dt.quarter,
        'year': dt.year
    }

    for name, values in datetime_features.items():
        features[name] = values

    return features


def extract_lag_features(
        series: pd.Series,
        config: dict,
        frequency) -> pd.DataFrame:
    """Generate lag features using config['lags']."""
    features = pd.DataFrame(index=series.index)
    try:
        for lag in config['lags']:
            try:
                features[f'lag_{lag}'] = series.shift(lag)
            except Exception as e:
                print(f"Error creating lag {lag}: {str(e)}")
    except Exception:
        try:
            lags = FREQ_CONFIG[frequency]['lags']
            for lag in lags:
                try:
                    features[f'lag_{lag}'] = series.shift(lag)
                except Exception as e:
                    print(f"Error creating lag {lag}: {str(e)}")
        except Exception as e:
            print("Error in returning lags: ", e)
    return features


def extract_rolling_features(
        series: pd.Series,
        config: dict,
        frequency) -> pd.DataFrame:
    """Calculate rolling window statistics."""
    features = pd.DataFrame(index=series.index)
    threshold = series.mean()
    try:
        for w in config['rolling_windows']:
            try:
                window = series.rolling(window=w)

                # Basic statistics
                features[f'rolling_mean_{w}'] = window.mean()
                features[f'rolling_std_{w}'] = window.std()
                features[f'rolling_min_{w}'] = window.min()
                features[f'rolling_max_{w}'] = window.max()

                # Quantiles
                q25 = window.quantile(0.25)
                q75 = window.quantile(0.75)
                features[f'rolling_quantile_25_{w}'] = q25
                features[f'rolling_quantile_75_{w}'] = q75
                features[f'rolling_iqr_{w}'] = q75 - q25

                # Other metrics
                features[f'rolling_median_{w}'] = window.median()
                features[f'rolling_sum_{w}'] = window.sum()
                features[f'rolling_skew_{w}'] = window.skew()
                features[f'rolling_kurtosis_{w}'] = window.kurt()

                # Difference features
                features[f'rolling_diff_{w}'] = series - series.shift(w)
                features[f'rolling_pct_change_{w}'] = series.pct_change(
                    periods=w)

                # Boolean counts
                features[f'rolling_count_above_mean_{w}'] = (
                    (series > threshold).rolling(window=w).sum()
                )
                # Dispersion metrics
                features[f'rolling_cv_{w}'] = features[f'rolling_std_{w}'] / \
                    features[f'rolling_mean_{w}']
                features[f'rolling_zscore_{w}'] = (
                    series - features[f'rolling_mean_{w}']) / features[f'rolling_std_{w}']

            except Exception as e:
                print(
                    f"Error creating rolling features for window {w}: {str(e)}")
    except Exception:
        for w in FREQ_CONFIG[frequency]['rolling_windows']:
            try:
                window = series.rolling(window=w)

                # Basic statistics
                features[f'rolling_mean_{w}'] = window.mean()
                features[f'rolling_std_{w}'] = window.std()
                features[f'rolling_min_{w}'] = window.min()
                features[f'rolling_max_{w}'] = window.max()

                # Quantiles
                q25 = window.quantile(0.25)
                q75 = window.quantile(0.75)
                features[f'rolling_quantile_25_{w}'] = q25
                features[f'rolling_quantile_75_{w}'] = q75
                features[f'rolling_iqr_{w}'] = q75 - q25

                # Other metrics
                features[f'rolling_median_{w}'] = window.median()
                features[f'rolling_sum_{w}'] = window.sum()
                features[f'rolling_skew_{w}'] = window.skew()
                features[f'rolling_kurtosis_{w}'] = window.kurt()

                # Difference features
                features[f'rolling_diff_{w}'] = series - series.shift(w)
                features[f'rolling_pct_change_{w}'] = series.pct_change(
                    periods=w)

                # Boolean counts
                features[f'rolling_count_above_mean_{w}'] = (
                    (series > threshold).rolling(window=w).sum()
                )
                # Dispersion metrics
                features[f'rolling_cv_{w}'] = features[f'rolling_std_{w}'] / \
                    features[f'rolling_mean_{w}']
                features[f'rolling_zscore_{w}'] = (
                    series - features[f'rolling_mean_{w}']) / features[f'rolling_std_{w}']

            except Exception as e:
                print(
                    f"Error creating rolling features for window {w}: {str(e)}")
    return features


def extract_seasonal_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """Perform seasonal decomposition."""
    features = pd.DataFrame(index=series.index)
    clean_series = series.dropna()

    try:
        decomposition = seasonal_decompose(
            clean_series, model='additive', period=config['seasonal_period']
        )
        features['trend_add'] = decomposition.trend
        features['seasonal_add'] = decomposition.seasonal
        features['residual_add'] = decomposition.resid

        try:
            decomposition = seasonal_decompose(
                clean_series,
                model='multiplicative',
                period=config['seasonal_period'])
            features['trend_mul'] = decomposition.trend
            features['seasonal_mul'] = decomposition.seasonal
            features['residual_mul'] = decomposition.resid
        except Exception as e:
            print(f"Error in multiplicative seasonal decomposition: {e}.")
            pass
    except Exception as e:
        print(f"Seasonal decomposition failed w/ infered period: {str(e)}")
        decomposition = seasonal_decompose(
            clean_series,
            model='additive',
            period=FREQ_CONFIG[frequency]['seasonal_period'])
        features['trend_add'] = decomposition.trend
        features['seasonal_add'] = decomposition.seasonal
        features['residual_add'] = decomposition.resid
        try:
            # Multiplicative decomposition
            decomposition = seasonal_decompose(
                clean_series,
                model='multiplicative',
                period=FREQ_CONFIG[frequency]['seasonal_period'])
            features['trend_mul'] = decomposition.trend
            features['seasonal_mul'] = decomposition.seasonal
            features['residual_mul'] = decomposition.resid
        except Exception as e:
            print(f"Error in multiplicative seasonal decomposition: {e}.")
            pass
    return features.reindex(series.index)


def extract_emd_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """Empirical Mode Decomposition features."""
    features = pd.DataFrame(index=series.index)

    try:
        emd = EMD()
        imfs = emd(series.values)

        for i in range(imfs.shape[0]):
            features[f'imf_{i+1}'] = np.nan
            valid_length = min(len(imfs[i]), len(features))
            features.iloc[-valid_length:,
                          features.columns.get_loc(f'imf_{i+1}')] = imfs[i][-valid_length:]

    except Exception as e:
        print(f"EMD failed: {str(e)}")

    return features


def extract_arima_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """ARIMA model residuals."""
    features = pd.DataFrame(index=series.index)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                series.dropna(),
                seasonal=False,
                suppress_warnings=True,
                error_action='ignore',
                n_jobs=CPU_COUNT,
            )
            features['arima_residual'] = np.nan
            valid_length = min(len(model.resid()), len(features))
            features.iloc[-valid_length:, 0] = model.resid()[-valid_length:]

    except Exception as e:
        print(f"ARIMA failed: {str(e)}")

    return features


def extract_tsfel_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """TSFEL features."""
    features = pd.DataFrame(index=series.index)

    try:
        cfg = tsfel.get_features_by_domain()
        tsfel_features = tsfel.time_series_features_extractor(
            cfg, series,
            window_size=config['tsfel_window'],
            overlap=0.5,
            n_jobs=CPU_COUNT  # Already parallelized at higher level
        )
        features = tsfel_features.add_prefix('tsfel_').reindex(series.index)

    except:
        print(f"TSFEL failed, using default config")
        cfg = tsfel.get_features_by_domain()
        tsfel_features = tsfel.time_series_features_extractor(
            cfg, series,
            window_size=FREQ_CONFIG[frequency]['tsfel_window'],
            overlap=0.5,
            n_jobs=CPU_COUNT  # Already parallelized at higher level
        )
        features = tsfel_features.add_prefix('tsfel_').reindex(series.index)
    return features


def extract_fft_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """FFT features."""
    features = pd.DataFrame(index=series.index)
    window_size = config['fft_window']

    try:
        fft_features = []
        for i in range(len(series) - window_size + 1):
            window = series.iloc[i:i + window_size]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            fft_features.append({
                'fft_peak_freq': np.argmax(amplitudes[1:window_size // 2]),
                'fft_peak_amp': np.max(amplitudes[1:window_size // 2])
            })

        fft_df = pd.DataFrame(fft_features,
                              index=series.index[window_size - 1:])
        features = features.join(fft_df, how='outer')

    except:
        print(f"FFT failed, using default config")
        window_size = FREQ_CONFIG[frequency]['fft_window']
        fft_features = []
        for i in range(len(series) - window_size + 1):
            window = series.iloc[i:i + window_size]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            fft_features.append({
                'fft_peak_freq': np.argmax(amplitudes[1:window_size // 2]),
                'fft_peak_amp': np.max(amplitudes[1:window_size // 2])
            })

        fft_df = pd.DataFrame(fft_features,
                              index=series.index[window_size - 1:])
        features = features.join(fft_df, how='outer')

    return features


def extract_wavelet_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """Wavelet transform features."""
    features = pd.DataFrame(index=series.index)
    scales = np.arange(1, 20)

    for wavelet in WAVELETS:
        try:
            coeffs, _ = pywt.cwt(series.values, scales, wavelet, method='fft')
            for i in range(coeffs.shape[0]):
                col_name = f'wavelet_{wavelet}_{i}'
                features[col_name] = np.nan
                valid_length = min(len(coeffs[i]), len(features))
                features.iloc[-valid_length:, features.columns.get_loc(
                    col_name)] = np.abs(coeffs[i][-valid_length:])

        except Exception as e:
            print(f"Wavelet '{wavelet}' failed: {str(e)}")

    return features


def extract_micro_fft_features(
        series: pd.Series,
        config: dict,
        frequency: str) -> pd.DataFrame:
    """Microstructure FFT features."""
    features = pd.DataFrame(index=series.index)
    micro_window = config['micro_window']

    try:
        micro_features = []
        for i in range(len(series) - micro_window + 1):
            window = series.iloc[i:i + micro_window]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            micro_features.append({
                'micro_fft_peak': np.argmax(amplitudes[1:micro_window // 2]),
                'micro_fft_amp': np.max(amplitudes[1:micro_window // 2])
            })

        micro_df = pd.DataFrame(micro_features,
                                index=series.index[micro_window - 1:])
        features = features.join(micro_df, how='outer')

    except Exception as e:
        print(f"Micro FFT failed: {str(e)}, using default config")
        micro_window = FREQ_CONFIG[frequency]['micro_window']
        micro_features = []
        for i in range(len(series) - micro_window + 1):
            window = series.iloc[i:i + micro_window]
            fft = np.fft.fft(window)
            amplitudes = np.abs(fft)
            micro_features.append({
                'micro_fft_peak': np.argmax(amplitudes[1:micro_window // 2]),
                'micro_fft_amp': np.max(amplitudes[1:micro_window // 2])
            })

        micro_df = pd.DataFrame(micro_features,
                                index=series.index[micro_window - 1:])
        features = features.join(micro_df, how='outer')

    return features

# ====================
# Main Featurization
# ====================


def auto_featurize(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Parallelized feature extraction pipeline.
    Returns DataFrame with original data and extracted features.
    """
    if frequency not in FREQ_MAP:
        raise ValueError(
            f"Invalid frequency. Choose from {list(FREQ_MAP.keys())}")

    if df.empty or len(df.columns) != 1:
        raise ValueError("DataFrame must have exactly one value column")

    value_col = df.columns[0]
    series = df[value_col].dropna()
    config = infer_time_series_params(series, frequency)

    # List of feature extraction functions
    feature_functions = [
        extract_datetime_features,
        extract_lag_features,
        extract_rolling_features,
        extract_seasonal_features,
        extract_emd_features,
        extract_arima_features,
        extract_tsfel_features,
        extract_fft_features,
        extract_wavelet_features,
        extract_micro_fft_features,
    ]

    # Execute all feature extractors in parallel
    features_list = Parallel(n_jobs=CPU_COUNT, prefer='processes')(
        delayed(func)(series, config, frequency) for func in feature_functions
    )

    # Combine all features with original data
    for feature_df in features_list:
        df = df.join(feature_df, how='left')

    # Final cleaning
    return df.ffill().bfill().dropna(how='all', axis=1)

def process_single_series(ts_df, freq, dataset_name):
    """Process one time series and save to CSV"""
    try:
        featurized_df = auto_featurize(ts_df, freq)
        series_name = ts_df.columns[0]
        dataset_dir = os.path.join("./data/monash", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        file_path = os.path.join(dataset_dir, f"{series_name}.csv")
        featurized_df.to_csv(file_path)
        print(f"Saved {series_name} to {file_path}")
        return True
    except Exception as e:
        print(f"Error in {ts_df.columns[0]}: {e}")
        return False

def process_dataset(data):
    """Process all series in a dataset"""
    s = perf_counter()
    name, df, freq = data['name'], data['df'], data['freq']
    print(f"Starting dataset: {name}")

    dataset_dir = os.path.join("./data/monash", name)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        all_series = list(prepare_time_series(df, freq))
    except Exception as e:
        print(f"Error preparing {name}: {e}")
        return (name, None)

    filtered_series = []
    for i in all_series:
        filename = f"{i.columns[0]}"
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            filtered_series.append(i)
    
    all_series = filtered_series

    if len(all_series) >= 50:
        all_series = random.sample(all_series, 50)

    results = Parallel(n_jobs=CPU_COUNT)(
        delayed(process_single_series)(ts_df, freq, name)
        for ts_df in all_series
    )

    success_count = sum(results)
    e = perf_counter()
    print(f"Finished {name} ({success_count}/{len(all_series)} succeeded)")
    return (name, (e - s) / 60)

def process_time_series(pickle_path="./data/monash/monash-df.pkl"):
    """Orchestrate processing of all datasets"""
    time_per_df = {}
    for data in yield_data(pickle_path):
        try:
            name, time_taken = process_dataset(data)
            if time_taken is not None:
                time_per_df[name] = time_taken
        except Exception as e:
            print(f"Error: {e}, skiping {data['name']}")
            continue
    return time_per_df

def main():
    total_start = perf_counter()
    time_per_df = process_time_series()  # Now properly defined
    total_end = perf_counter()
    
    print(f"\nTotal time: {(total_end - total_start)/60:.2f} mins")
    print("Per-dataset times:", time_per_df)
