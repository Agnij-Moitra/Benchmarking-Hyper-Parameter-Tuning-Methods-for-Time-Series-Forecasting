import numpy as np
import pickle
import pandas as pd


def yield_data(pickle_file_path="./data/monash/monash-df.pkl"):
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
                    obj = pickle.load(f)
                    items = list(obj.items())[0]
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


def prepare_time_series(df, frequency):
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


def calculate_sma(data: pd.Series, n: int) -> pd.Series:
    """
    ## Calculate Simple Moving Average (SMA) for a pandas Series.

    ### Args:
        - data (`pd.Series`): Input pandas Series with numerical data
        - n (`int`): Window size for moving average

    ### Returns:
        - `pd.Series`: Series containing SMA values
    """
    return data.rolling(window=n).mean()


def calculate_ema(data: pd.Series, n: int) -> pd.Series:
    """
    ## Calculate Exponential Moving Average (EMA) for a pandas Series.

    ### Args:
        - data (`pd.Series`): Input pandas Series with numerical data
        - n (`int`): Span (number of periods) for EMA calculation

    ### Returns:
        - `pd.Series`: Series containing EMA values
    """
    return data.ewm(span=n, adjust=False).mean()


def calculate_wma(data: pd.Series, n: int) -> pd.Series:
    """
    ## Calculate Weighted Moving Average (WMA) for a pandas Series.

    ### Args:
        - data (`pd.Series`): Input pandas Series with numerical data
        - n (`int`): Window size for moving average

    ### Returns:
        - `pd.Series`: Series containing WMA values
    """
    weights = np.arange(1, n + 1)
    return data.rolling(window=n).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calculate_hma(data: pd.Series, n: int) -> pd.Series:
    """
    ## Calculate Hull Moving Average (HMA) for a pandas Series.

    ### Args:
        - data (`pd.Series`): Input pandas Series with numerical data
        - n (`int`): Period for HMA calculation

    ### Returns:
        - `pd.Series`: Series containing HMA values
    """
    sqrt_n = int(np.sqrt(n))
    weights = np.arange(1, sqrt_n + 1)
    return (2 * data.rolling(window=n // 2).apply(
        lambda x: np.dot(x, np.arange(1, n // 2 + 1)) /
        np.sum(np.arange(1, n // 2 + 1)),
        raw=True
    ) - data.rolling(window=n).apply(
        lambda x: np.dot(x, np.arange(1, n + 1)) / np.sum(np.arange(1, n + 1)),
        raw=True
    )).rolling(window=sqrt_n).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def calculate_kama(
        data: pd.Series,
        n: int = 10,
        fast_sc: int = 5,
        slow_sc: int = 30) -> pd.Series:
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Lookback period for efficiency ratio (default: 10)
        fast_sc (int): Fast smoothing constant period (default: 5)
        slow_sc (int): Slow smoothing constant period (default: 30)

    Returns:
        pd.Series: Series containing KAMA values
    """
    fast_sc = 2 / (fast_sc + 1)
    slow_sc = 2 / (slow_sc + 1)
    sc = (abs(data - data.shift(n)) / abs(data - data.shift(1)
                                          ).rolling(n).sum() * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=data.index, dtype=float)
    kama.iloc[n] = data.iloc[n]
    for i in range(n + 1, len(data)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * \
            (data.iloc[i] - kama.iloc[i - 1])
    return kama


def calculate_dma(data: pd.Series, n: int, displacement: int) -> pd.Series:
    """
    Calculate Displaced Moving Average (DMA) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Window size for moving average
        displacement (int): Number of periods to shift (positive for forward, negative for backward)

    Returns:
        pd.Series: Series containing DMA values
    """
    return data.rolling(window=n).mean().shift(periods=displacement)


def calculate_dema(data: pd.Series, n: int) -> pd.Series:
    """
    Calculate Double Exponential Moving Average (DEMA) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Span (number of periods) for EMA calculation

    Returns:
        pd.Series: Series containing DEMA values
    """
    ema = data.ewm(span=n, adjust=False).mean()
    return 2 * ema - ema.ewm(span=n, adjust=False).mean()


def calculate_vidya(
        data: pd.Series,
        cmo_period: int = 9,
        ema_period: int = 9) -> pd.Series:
    """
    Calculate Volatility Index Dynamic Average (VIDYA) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        cmo_period (int): Period for Chande Momentum Oscillator (default: 9)
        ema_period (int): Period for EMA smoothing factor (default: 9)

    Returns:
        pd.Series: Series containing VIDYA values
    """
    diff = data.diff()
    up_sum = diff.where(diff > 0, 0).rolling(window=cmo_period).sum()
    down_sum = abs(diff.where(diff < 0, 0)).rolling(window=cmo_period).sum()
    abs_cmo = abs((up_sum - down_sum) / (up_sum + down_sum))
    f = 2 / (ema_period + 1)
    vidya = pd.Series(index=data.index, dtype=float)
    vidya.iloc[cmo_period] = data.iloc[cmo_period]
    for i in range(cmo_period + 1, len(data)):
        vidya.iloc[i] = data.iloc[i] * f * abs_cmo.iloc[i] + \
            vidya.iloc[i - 1] * (1 - f * abs_cmo.iloc[i])
    return vidya


def calculate_vma(data: pd.Series, n: int, vol_period: int = 20) -> pd.Series:
    """
    Calculate Volatility Moving Average (V-MA) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Base period for moving average
        vol_period (int): Period for volatility calculation (default: 20)

    Returns:
        pd.Series: Series containing V-MA values
    """
    volatility = data.rolling(window=vol_period).std()
    vol_weights = volatility / volatility.mean()
    vma = pd.Series(index=data.index, dtype=float)
    for i in range(n - 1, len(data)):
        window = data.iloc[i - n + 1:i + 1]
        weights = vol_weights.iloc[i - n + 1:i + 1]
        if len(window) == n and not weights.isna().any():
            weighted_avg = np.average(window, weights=weights)
            vma.iloc[i] = weighted_avg
    return vma


def calculate_rsi(data: pd.Series, n: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for RSI calculation (default: 14)

    Returns:
        pd.Series: Series containing RSI values (0-100)
    """
    delta = data.diff()
    return 100 - (100 / (1 + delta.where(delta > 0,
                                         0).rolling(window=n).mean() / -delta.where(delta < 0,
                                                                                    0).rolling(window=n).mean()))


def calculate_roc(data: pd.Series, n: int) -> pd.Series:
    """
    Calculate Rate of Change (ROC) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Lookback period for ROC calculation

    Returns:
        pd.Series: Series containing ROC values (as percentage)
    """
    series_n = data.shift(n)
    return ((data - series_n) / series_n) * 100


def calculate_momentum(data: pd.Series, n: int) -> pd.Series:
    """
    Calculate Momentum Indicator for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Lookback period for momentum calculation

    Returns:
        pd.Series: Series containing momentum values
    """
    return data - data.shift(n)


def calculate_stochastic_rsi_k(
        data: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3) -> pd.Series:
    """
    Calculate Stochastic RSI for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        rsi_period (int): Period for RSI calculation (default: 14)
        stoch_period (int): Period for Stochastic calculation (default: 14)
        smooth_k (int): Smoothing period for %K line (default: 3)

    Returns:
        pd.Series: Series containing Stochastic RSI %K values
    """
    rsi = calculate_rsi(data=data, n=rsi_period)
    lowest_rsi = rsi.rolling(window=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period).max()
    return (((rsi - lowest_rsi) / (highest_rsi - lowest_rsi))
            * 100).rolling(window=smooth_k).mean()


def calculate_stochastic_rsi_k(
        data: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3) -> pd.Series:
    """
    Calculate Stochastic RSI for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        rsi_period (int): Period for RSI calculation (default: 14)
        stoch_period (int): Period for Stochastic calculation (default: 14)
        smooth_k (int): Smoothing period for %K line (default: 3)
        smooth_d (int): Smoothing period for %D line (default: 3)

    Returns:
        pd.Series: Series containing Stochastic RSI %K values
    """
    rsi = calculate_rsi(data=data, n=rsi_period)
    lowest_rsi = rsi.rolling(window=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period).max()
    return (((rsi - lowest_rsi) / (highest_rsi - lowest_rsi)) * \
            100).rolling(window=smooth_k).mean().rolling(window=smooth_d).mean()


def calculate_ppo(
        data: pd.Series,
        short_period: int = 12,
        long_period: int = 26) -> pd.Series:
    """
    Calculate Percentage Price Oscillator (PPO) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        short_period (int): Period for short-term EMA (default: 12)
        long_period (int): Period for long-term EMA (default: 26)

    Returns:
        pd.Series: Series containing PPO values (as percentage)
    """
    ema_long = data.ewm(span=long_period, adjust=False).mean()
    return ((data.ewm(span=short_period, adjust=False).mean() -
            ema_long) / ema_long) * 100


def calculate_macd(
        data: pd.Series,
        short_period: int = 12,
        long_period: int = 26) -> pd.Series:
    """
    Calculate Moving Average Convergence Divergence (MACD) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        short_period (int): Period for short-term EMA (default: 12)
        long_period (int): Period for long-term EMA (default: 26)

    Returns:
        pd.Series: Series containing MACD values
    """
    return data.ewm(span=short_period, adjust=False).mean() - \
        data.ewm(span=long_period, adjust=False).mean()


def calculate_macd_histogram(
        data: pd.Series,
        short_period: int = 12,
        long_period: int = 26,
        signal_period: int = 9) -> pd.Series:
    """
    Calculate MACD Histogram for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        short_period (int): Period for short-term EMA (default: 12)
        long_period (int): Period for long-term EMA (default: 26)
        signal_period (int): Period for signal line EMA (default: 9)

    Returns:
        pd.Series: Series containing MACD Histogram values
    """
    macd = calculate_macd(
        data=data, short_period=short_period, long_period=long_period)
    return macd - macd.ewm(span=signal_period, adjust=False).mean()


def calculate_trix(data: pd.Series, n: int = 15) -> pd.Series:
    """
    Calculate TRIX (Triple Exponential Average) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for EMA calculations (default: 15)

    Returns:
        pd.Series: Series containing TRIX values (as percentage)
    """
    ema3 = data.ewm(
        span=n,
        adjust=False).mean().ewm(
        span=n,
        adjust=False).mean().ewm(
            span=n,
        adjust=False).mean()
    ema3_previous = ema3.shift(1)
    return 100 * (ema3 - ema3_previous) / ema3_previous


def calculate_rmi(
        data: pd.Series,
        n: int = 14,
        momentum_period: int = 5) -> pd.Series:
    """
    Calculate Relative Momentum Index (RMI) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for average gains/losses (default: 14)
        momentum_period (int): Lookback period for momentum (default: 5)

    Returns:
        pd.Series: Series containing RMI values (0-100)
    """
    momentum = data - data.shift(momentum_period)
    return 100 - (100 / (1 + momentum.where(momentum > 0,
                                            0).rolling(window=n).mean() / -momentum.where(momentum < 0,
                                                                                          0).rolling(window=n).mean()))


def calculate_pmo(data: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Price Momentum Oscillator (PMO) using the standard method for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data (e.g., closing prices).
        n (int): Period for Rate of Change (ROC) and EMA calculation (default: 20).

    Returns:
        pd.Series: Series containing PMO values.
    """
    roc = (data - data.shift(n)) / data.shift(n) * 100
    return 10 * roc.ewm(span=n,
                        adjust=False).mean().ewm(span=n,
                                                 adjust=False).mean()


def calculate_coppock(
        data: pd.Series,
        roc_short: int = 11,
        roc_long: int = 14,
        wma_period: int = 10) -> pd.Series:
    """
    Calculate Coppock Curve for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        roc_short (int): Period for short-term ROC (default: 11)
        roc_long (int): Period for long-term ROC (default: 14)
        wma_period (int): Period for WMA calculation (default: 10)

    Returns:
        pd.Series: Series containing Coppock Curve values
    """
    roc_sum = ((data - data.shift(roc_short)) / data.shift(roc_short)) * \
        100 + ((data - data.shift(roc_long)) / data.shift(roc_long)) * 100
    weights = np.arange(1, wma_period + 1)
    return roc_sum.rolling(window=wma_period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calculate_tii(data: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Trend Intensity Index (TII) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for SMA and deviation calculations (default: 20)

    Returns:
        pd.Series: Series containing TII values (0-100)
    """
    deviations = data - data.rolling(window=n).mean()
    sum_pos_dev = deviations.where(deviations > 0, 0).rolling(window=n).sum()
    sum_neg_dev = -deviations.where(deviations < 0, 0).rolling(window=n).sum()
    return 100 * (sum_pos_dev - sum_neg_dev) / (sum_pos_dev + sum_neg_dev)


def calculate_dpo(data: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Detrended Price Oscillator (DPO) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for SMA calculation (default: 20)

    Returns:
        pd.Series: Series containing DPO values
    """
    return data - data.rolling(window=n).mean().shift(-(n // 2 + 1))


def calculate_regression_slope(data: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate the slope of the Regression Curve for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for regression calculation (default: 20)

    Returns:
        pd.Series: Series containing regression slope values
    """
    # Initialize output series
    slopes = pd.Series(index=data.index, dtype=float)
    x = np.arange(n)
    for i in range(n - 1, len(data)):
        y = data.iloc[i - n + 1:i + 1].values
        if len(y) == n and not np.any(np.isnan(y)):
            slopes.iloc[i] = np.polyfit(x, y, 1)[0]
    return slopes


def calculate_ulcer_index(data: pd.Series, n: int = 14) -> pd.Series:
    """
    Calculate Ulcer Index for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for Ulcer Index calculation (default: 14)

    Returns:
        pd.Series: Series containing Ulcer Index values
    """
    highest_high = data.rolling(window=n).max()
    drawdown_squared = ((data - highest_high) / highest_high) ** 2
    return np.sqrt(drawdown_squared.rolling(window=n).sum() / n)


def calculate_vhf(data: pd.Series, n: int = 28) -> pd.Series:
    """
    Calculate Vertical Horizontal Filter (VHF) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for VHF calculation (default: 28)

    Returns:
        pd.Series: Series containing VHF values
    """
    return (data.rolling(window=n).max() - data.rolling(window=n).min()
            ) / abs(data.diff()).rolling(window=n).sum()


def calculate_aroon_up(data: pd.Series, n: int = 25) -> pd.Series:
    """
    Calculate Aroon Up Indicator for a pandas Series, adapted for peak .

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for Aroon Up calculation (default: 25)

    Returns:
        pd.Series: Series containing Aroon Up values
    """
    aroon_up = pd.Series(index=data.index, dtype=float)
    for i in range(n - 1, len(data)):
        aroon_up.iloc[i] = (
            (1 + np.argmax(data.iloc[i - n + 1:i + 1])) / n) * 100
    return aroon_up


def calculate_aroon_down(data: pd.Series, n: int = 25) -> pd.Series:
    """
    Calculate Aroon Down Indicator for a pandas Series, adapted for peak .

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for Aroon Down calculation (default: 25)

    Returns:
        pd.Series: Series containing Aroon Down values
    """
    aroon_down = pd.Series(index=data.index, dtype=float)
    for i in range(n - 1, len(data)):
        aroon_down.iloc[i] = (
            (1 + np.argmin(data.iloc[i - n + 1:i + 1])) / n) * 100
    return aroon_down


def calculate_aroon_oscillator(data: pd.Series, n: int = 25) -> pd.Series:
    """
    Calculate Aroon Oscillator for a pandas Series, adapted for peak .

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for Aroon calculation (default: 25)

    Returns:
        pd.Series: Series containing Aroon Oscillator values
    """
    aroon_up = pd.Series(index=data.index, dtype=float)
    aroon_down = pd.Series(index=data.index, dtype=float)
    for i in range(n - 1, len(data)):
        window = data.iloc[i - n + 1:i + 1]
        aroon_up.iloc[i] = (
            (1 + np.argmax(window)) / n) * 100
        aroon_down.iloc[i] = (
            (1 + np.argmin(window)) / n) * 100
    return aroon_up - aroon_down


def calculate_vortex_plus(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 14) -> pd.Series:
    """
    Calculate Vortex Plus (VI+) Indicator for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high  data
        data_low (pd.Series): Input pandas Series with low  data
        n (int): Period for Vortex Plus calculation (default: 14)

    Returns:
        pd.Series: Series containing Vortex Plus (VI+) values
    """
    return abs(data_high - data_low.shift(1)).rolling(window=n).sum() / pd.concat([
        (data_high - data_low),
        abs(data_high - data_high.shift(1)),
        abs(data_low - data_low.shift(1))
    ], axis=1).max(axis=1).rolling(window=n).sum()


def calculate_vortex_minus(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 14) -> pd.Series:
    """
    Calculate Vortex Minus (VI-) Indicator for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high data
        data_low (pd.Series): Input pandas Series with low data
        n (int): Period for Vortex Minus calculation (default: 14)

    Returns:
        pd.Series: Series containing Vortex Minus (VI-) values
    """
    return abs(data_low - data_high.shift(1)).rolling(window=n).sum() / pd.concat([
        (data_high - data_low),
        abs(data_high - data_high.shift(1)),
        abs(data_low - data_low.shift(1))
    ], axis=1).max(axis=1).rolling(window=n).sum()


def calculate_mass_index(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 25,
        ema_period: int = 9) -> pd.Series:
    """
    Calculate Mass Index for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high  data
        data_low (pd.Series): Input pandas Series with low  data
        n (int): Period for Mass Index calculation (default: 25)
        ema_period (int): Period for EMA calculations (default: 9)

    Returns:
        pd.Series: Series containing Mass Index values
    """
    hl_range = data_high - data_low
    ema1 = hl_range.ewm(span=ema_period, adjust=False).mean()
    ratio = ema1 / ema1.ewm(span=ema_period, adjust=False).mean()
    return ratio.rolling(window=n).sum()


def calculate_mass_index(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 25,
        ema_period: int = 9) -> pd.Series:
    """
    Calculate Mass Index for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high data
        data_low (pd.Series): Input pandas Series with low data
        n (int): Period for Mass Index summation (default: 25)
        ema_period (int): Period for EMA calculations (default: 9)

    Returns:
        pd.Series: Series containing Mass Index values
    """
    hl_range = data_high - data_low
    ema1 = hl_range.ewm(span=ema_period, adjust=False).mean()
    ratio = ema1 / ema1.ewm(span=ema_period, adjust=False).mean()
    return ratio.rolling(window=n).sum()


def calculate_clv(
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series) -> pd.Series:
    """
    Calculate Close Location Value (CLV) for pandas Series of high, low, and close data.

    Args:
        data_high (pd.Series): Input pandas Series with high data
        data_low (pd.Series): Input pandas Series with low data
        data_close (pd.Series): Input pandas Series with close data

    Returns:
        pd.Series: Series containing CLV values
    """
    denominator = data_high - data_low
    return (2 * data_close - data_low - data_high) / \
        denominator.where(denominator != 0, 1)


def calculate_bop(
        data_open: pd.Series,
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series) -> pd.Series:
    """
    Calculate Balance of Power (BOP) for pandas Series of open, close, high, and low data.

    Args:
        data_open (pd.Series): Input pandas Series with open data
        data_close (pd.Series): Input pandas Series with close data
        data_high (pd.Series): Input pandas Series with high data
        data_low (pd.Series): Input pandas Series with low data

    Returns:
        pd.Series: Series containing BOP values
    """
    denominator = data_high - data_low
    return (data_close - data_open) / denominator.where(denominator != 0, 1)


def calculate_bull_power(data_high: pd.Series, n: int = 13) -> pd.Series:
    """
    Calculate Elder Ray Bull Power for pandas Series of high data.

    Args:
        data_high (pd.Series): Input pandas Series with high data
        n (int): Period for EMA calculation (default: 13)

    Returns:
        pd.Series: Series containing Bull Power values
    """
    return data_high - data_high.ewm(span=n, adjust=False).mean()


def calculate_bear_power(data_low: pd.Series, n: int = 13) -> pd.Series:
    """
    Calculate Elder Ray Bear Power for pandas Series of low data.

    Args:
        data_low (pd.Series): Input pandas Series with low data
        n (int): Period for EMA calculation (default: 13)

    Returns:
        pd.Series: Series containing Bear Power values
    """
    return data_low - data_low.ewm(span=n, adjust=False).mean()


def calculate_heikin_ashi_close(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series) -> pd.Series:
    """Calculate Heikin Ashi Close."""
    return (data_open + data_high + data_low + data_close) / 4


def calculate_heikin_ashi_open(
        data_open: pd.Series,
        ha_close_series: pd.Series) -> pd.Series:
    """Calculate Heikin Ashi Open."""
    ha_open_series = pd.Series(index=data_open.index, dtype=float)
    for i in range(len(data_open)):
        if i == 0:
            ha_open_series.iloc[i] = data_open.iloc[i]
        else:
            ha_open_series.iloc[i] = (
                ha_open_series.iloc[i - 1] + ha_close_series.iloc[i - 1]) / 2
    return ha_open_series


def calculate_heikin_ashi_high(
        data_high: pd.Series,
        ha_open_series: pd.Series,
        ha_close_series: pd.Series) -> pd.Series:
    """Calculate Heikin Ashi High."""
    return pd.concat([data_high, ha_open_series,
                     ha_close_series], axis=1).max(axis=1)


def calculate_heikin_ashi_low(
        data_low: pd.Series,
        ha_open_series: pd.Series,
        ha_close_series: pd.Series) -> pd.Series:
    """Calculate Heikin Ashi Low."""
    return pd.concat(
        [data_low, ha_open_series, ha_close_series], axis=1).min(axis=1)


def swing_index(
    data_open: pd.Series,
    data_high: pd.Series,
    data_low: pd.Series,
    data_close: pd.Series
) -> pd.Series:
    """
    Calculate the Swing Index (SI) using Welles Wilder's original formula.

    Parameters:
        data_open (pd.Series): Series of open prices
        data_high (pd.Series): Series of high prices
        data_low (pd.Series): Series of low prices
        data_close (pd.Series): Series of close prices

    Returns:
        pd.Series: Swing Index time series
    """
    c_prev = data_close.shift(1)
    o_prev = data_open.shift(1)
    tr = pd.concat([
        (data_high - data_low).abs(),
        (data_high - c_prev).abs(),
        (data_low - c_prev).abs()
    ], axis=1).max(axis=1)
    si = 50 * (
        (data_close - c_prev)
        + 0.5 * (data_close - data_open)
        + 0.25 * (c_prev - o_prev)
    ) / tr + 0.5 * (data_close - c_prev).abs() + 0.25 * (c_prev - o_prev).abs() * (pd.concat([
        data_high - c_prev,
        c_prev - data_low
    ], axis=1).max(axis=1) / tr)
    return si.replace([np.inf, -np.inf], np.nan).fillna(0)


def accumulated_swing_index(
    data_open: pd.Series,
    data_high: pd.Series,
    data_low: pd.Series,
    data_close: pd.Series
) -> pd.Series:
    """
    Calculate the Accumulated Swing Index (ASI).

    Parameters:
        data_open (pd.Series): Open prices
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices

    Returns:
        pd.Series: Accumulated Swing Index time series
    """
    return swing_index(data_open, data_high, data_low, data_close).cumsum()


def typical_price(
    data_high: pd.Series,
    data_low: pd.Series,
    data_close: pd.Series
) -> pd.Series:
    """
    Calculate the Typical Price.

    Parameters:
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices

    Returns:
        pd.Series: Typical Price time series
    """
    return (data_high + data_low + data_close) / 3


def weighted_close(
    data_high: pd.Series,
    data_low: pd.Series,
    data_close: pd.Series
) -> pd.Series:
    """
    Calculate the Weighted Close.

    Parameters:
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices

    Returns:
        pd.Series: Weighted Close time series
    """
    return (data_high + data_low + 2 * data_close) / 4


def percent_k(data_high: pd.Series,
              data_low: pd.Series,
              data_close: pd.Series,
              period: int = 14) -> pd.Series:
    """
    Calculate the %K of the Stochastic Oscillator.

    Parameters:
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices
        period (int): Look-back period for %K calculation (default is 14)

    Returns:
        pd.Series: %K values
    """
    lowest_low = data_low.rolling(window=period).min()
    return ((data_close - lowest_low) /
            (data_high.rolling(window=period).max() - lowest_low)) * 100


def percent_d(percent_k: pd.Series, smooth_period: int = 10) -> pd.Series:
    """
    Calculate the %D of the Stochastic Oscillator (SMA of %K).

    Parameters:
        percent_k (pd.Series): The %K values
        smooth_period (int): Period for smoothing %K to get %D (default is 10)

    Returns:
        pd.Series: %D values (SMA of %K)
    """
    return percent_k.rolling(window=smooth_period).mean()


def williams_r(data_high: pd.Series,
               data_low: pd.Series,
               data_close: pd.Series,
               period: int = 14) -> pd.Series:
    """
    Calculate the Williams %R indicator.

    Parameters:
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices
        period (int): Look-back period (default is 14)

    Returns:
        pd.Series: Williams %R values
    """
    highest_high = data_high.rolling(window=period).max()
    return ((highest_high - data_close) / (highest_high -
            data_low.rolling(window=period).min())) * -100


def commodity_channel_index(data_high: pd.Series,
                            data_low: pd.Series,
                            data_close: pd.Series,
                            period: int = 20) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI).

    Parameters:
        data_high (pd.Series): High prices
        data_low (pd.Series): Low prices
        data_close (pd.Series): Close prices
        period (int): Look-back period for CCI calculation (default is 20)

    Returns:
        pd.Series: CCI values
    """
    typical_price = (data_high + data_low + data_close) / 3
    return (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * \
            typical_price.rolling(window=period).apply(lambda x: pd.Series(x).mad(), raw=False))


def chande_momentum_oscillator(
        data_close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Calculate the Chande Momentum Oscillator (CMO).

    Parameters:
        data_close (pd.Series): Series of closing prices
        period (int): Look-back period for CMO calculation (default is 14)

    Returns:
        pd.Series: CMO values
    """
    delta = data_close.diff()
    sum_gain = delta.where(
        delta > 0,
        0.0).rolling(
        window=period,
        min_periods=period).sum()
    sum_loss = -delta.where(delta < 0,
                            0.0).rolling(window=period,
                                         min_periods=period).sum()
    return 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)


def ultimate_oscillator(high: pd.Series,
                        low: pd.Series,
                        close: pd.Series,
                        short_period: int = 7,
                        medium_period: int = 14,
                        long_period: int = 28) -> pd.Series:
    """
    Calculate the Ultimate Oscillator (UO).

    Parameters:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        short_period (int): Short-term period (default is 7)
        medium_period (int): Medium-term period (default is 14)
        long_period (int): Long-term period (default is 28)

    Returns:
        pd.Series: Ultimate Oscillator values
    """
    prior_close = close.shift(1)
    bp = close - pd.concat([low, prior_close], axis=1).min(axis=1)
    tr = pd.concat([high, prior_close], axis=1).max(axis=1) - \
        pd.concat([low, prior_close], axis=1).min(axis=1)
    return 100 * ((4 * bp.rolling(window=short_period).sum() / tr.rolling(window=short_period).sum()) + (2 * bp.rolling(window=medium_period).sum() /
                  tr.rolling(window=medium_period).sum()) + (bp.rolling(window=long_period).sum() / tr.rolling(window=long_period).sum())) / 7


def stochastic_rsi(data_close: pd.Series,
                   rsi_period: int = 14,
                   stoch_period: int = 14) -> pd.Series:
    """
    Calculate the Stochastic RSI (StochRSI).

    Parameters:
        data_close (pd.Series): Series of closing prices.
        rsi_period (int): Period for RSI calculation (default is 14).
        stoch_period (int): Period for StochRSI calculation (default is 14).

    Returns:
        pd.Series: StochRSI values.
    """
    delta = data_close.diff()
    rsi = 100 - (100 / (1 + delta.where(delta > 0,
                                        0.0).rolling(window=rsi_period,
                 min_periods=rsi_period).mean() / (-delta.where(delta < 0,
                                                   0.0).rolling(window=rsi_period,
                                                                min_periods=rsi_period).mean())))
    min_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    return (rsi - min_rsi) / (rsi.rolling(window=stoch_period,
                                          min_periods=stoch_period).max() - min_rsi)


def derivative_oscillator(data_close: pd.Series,
                          rsi_period: int = 14,
                          ema1_period: int = 5,
                          ema2_period: int = 3,
                          sma_period: int = 9) -> pd.Series:
    """
    Calculate the Derivative Oscillator.

    Parameters:
        data_close (pd.Series): Series of closing prices.
        rsi_period (int): Period for RSI calculation (default is 14).
        ema1_period (int): Period for the first EMA smoothing (default is 5).
        ema2_period (int): Period for the second EMA smoothing (default is 3).
        sma_period (int): Period for the SMA signal line (default is 9).

    Returns:
        pd.Series: Derivative Oscillator values.
    """
    # Calculate price differences
    delta = data_close.diff()
    rsi = 100 - (100 / (1 + delta.where(delta > 0,
                                        0.0).rolling(window=rsi_period,
                 min_periods=rsi_period).mean() / (-delta.where(delta < 0,
                                                   0.0).rolling(window=rsi_period,
                                                                min_periods=rsi_period).mean())))
    ema2 = rsi.ewm(
        span=ema1_period,
        adjust=False).mean().ewm(
        span=ema2_period,
        adjust=False).mean()
    return ema2 - ema2.rolling(window=sma_period,
                               min_periods=sma_period).mean()


def awesome_oscillator(high: pd.Series,
                       low: pd.Series,
                       short_period: int = 5,
                       long_period: int = 34) -> pd.Series:
    """
    Calculate the Awesome Oscillator (AO).

    Parameters:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        short_period (int): Period for the short-term SMA (default is 5).
        long_period (int): Period for the long-term SMA (default is 34).

    Returns:
        pd.Series: Awesome Oscillator values.
    """
    median_price = (high + low) / 2
    return median_price.rolling(window=short_period, min_periods=short_period).mean(
    ) - median_price.rolling(window=long_period, min_periods=long_period).mean()


def fib_0(high: pd.Series, low: pd.Series) -> pd.Series:
    return high


def fib_23_6(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - 0.236 * (high - low)


def fib_38_2(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - 0.382 * (high - low)


def fib_50_0(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - 0.5 * (high - low)


def fib_61_8(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - 0.618 * (high - low)


def fib_78_6(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - 0.786 * (high - low)


def fib_100(high: pd.Series, low: pd.Series) -> pd.Series:
    return low


def fib_161_8(high: pd.Series, low: pd.Series) -> pd.Series:
    return high + 0.618 * (high - low)


def fib_261_8(high: pd.Series, low: pd.Series) -> pd.Series:
    return high + 1.618 * (high - low)


def fib_423_6(high: pd.Series, low: pd.Series) -> pd.Series:
    return high + 2.236 * (high - low)


def fib_533_6(high: pd.Series, low: pd.Series) -> pd.Series:
    return high + 3.236 * (high - low)


def fib_1000(high: pd.Series, low: pd.Series) -> pd.Series:
    return high + 9 * (high - low)


def pivot_point(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series) -> pd.Series:
    """Calculate Pivot Point (PP)."""
    return (high + low + close) / 3


def support_1(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate First Support (S1)."""
    pp = pivot_point(high, low, close)
    return (2 * pp) - high


def resistance_1(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series) -> pd.Series:
    """Calculate First Resistance (R1)."""
    pp = pivot_point(high, low, close)
    return (2 * pp) - low


def support_2(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Second Support (S2)."""
    pp = pivot_point(high, low, close)
    return pp - (high - low)


def resistance_2(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series) -> pd.Series:
    """Calculate Second Resistance (R2)."""
    pp = pivot_point(high, low, close)
    return pp + (high - low)


def support_3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Third Support (S3)."""
    pp = pivot_point(high, low, close)
    return low - 2 * (high - pp)


def resistance_3(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series) -> pd.Series:
    """Calculate Third Resistance (R3)."""
    pp = pivot_point(high, low, close)
    return high + 2 * (pp - low)


def find_wave_1_points(df: pd.DataFrame):
    """
    Identify the start and end points of Wave 1 based on local minima and maxima.
    The start of Wave 1 is a local minimum, and the end is a local maximum.
    """
    min_idx = df['Low'].idxmin()
    max_idx = df['High'].idxmax()
    if min_idx < max_idx:
        wave_1_start = min_idx
        wave_1_end = max_idx
    else:
        max_idx = df.loc[min_idx:]['High'].idxmax()
        wave_1_start = min_idx
        wave_1_end = max_idx

    return wave_1_start, wave_1_end


def fibonacci_retracement_wave2(start: pd.Series, end: pd.Series) -> pd.Series:
    """
    Calculate Fibonacci retracement levels for Wave 2 (correction of Wave 1).
    """
    diff = end - start
    retracements = {
        '23.6%': end - 0.236 * diff,
        '38.2%': end - 0.382 * diff,
        '50.0%': end - 0.5 * diff,
        '61.8%': end - 0.618 * diff,
        '100%': start
    }
    return pd.Series(retracements)


def fibonacci_extension_wave3(start: pd.Series, end: pd.Series) -> pd.Series:
    """
    Calculate Fibonacci extension levels for Wave 3 (extension of Wave 1).
    """
    diff = end - start
    extensions = {
        '161.8%': end + 1.618 * diff,
        '261.8%': end + 2.618 * diff,
        '423.6%': end + 4.236 * diff
    }
    return pd.Series(extensions)


def calculate_elliott_wave_2(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Fibonacci retracement levels for Wave 2 based on Wave 1.
    """
    wave_1_start, wave_1_end = find_wave_1_points(df)
    return fibonacci_retracement_wave2(
        df['Low'][wave_1_start],
        df['High'][wave_1_end])


def calculate_elliott_wave_3(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Fibonacci extension levels for Wave 3 based on Wave 1.
    """
    wave_1_start, wave_1_end = find_wave_1_points(df)
    return fibonacci_extension_wave3(
        df['Low'][wave_1_start],
        df['High'][wave_1_end])


def hurst_exponent(df: pd.Series) -> float:
    """
    Estimate the Hurst Exponent (H) using rescaled range analysis (R/S).
    H > 0.5 indicates a persistent, trending time series.
    H < 0.5 indicates a mean-reverting time series.

    Args:
        df (pd.Series): Time series data.

    Returns:
        float: The estimated Hurst Exponent.
    """
    cumulative_deviation = np.cumsum(df - df.mean())
    return np.log((np.max(cumulative_deviation) -
                  np.min(cumulative_deviation)) / np.std(df)) / np.log(len(df))


def range_indicator(
        data_high: pd.Series,
        data_low: pd.Series,
        N: int = 14) -> pd.Series:
    """
    Calculate the Range Indicator (Closing Price Range) which is the difference
    between the maximum high and minimum low prices over a period of N.

    Args:
        data_high (pd.Series): A pandas Series of high prices.
        data_low (pd.Series): A pandas Series of low prices.
        N (int, optional): The number of periods over which the range is calculated. Default is 14.

    Returns:
        pd.Series: A pandas Series of the Range Indicator values.
    """
    return data_high.rolling(window=N).max() - data_low.rolling(window=N).min()


def calculate_iqr(data: pd.Series, N: int = 14) -> pd.Series:
    """
    Calculate the Interquartile Range (IQR) over a rolling window of N periods.

    Args:
        data (pd.Series): Input pandas Series with numerical data.
        N (int): Period for rolling calculation (default: 14).

    Returns:
        pd.Series: Series containing IQR values.
    """
    return data.rolling(window=N).quantile(
        0.75) - data.rolling(window=N).quantile(0.25)


def calculate_cv(data: pd.Series, N: int = 14) -> pd.Series:
    """
    Calculate the Coefficient of Variation (CV) over a rolling window of N periods.

    Args:
        data (pd.Series): Input pandas Series with numerical data.
        N (int): Period for rolling calculation (default: 14).

    Returns:
        pd.Series: Series containing CV values.
    """
    return (data.rolling(window=N).std() / data.rolling(window=N).mean()) * 100


def calculate_mad(data: pd.Series, N: int = 14) -> pd.Series:
    """
    Calculate the Median Absolute Deviation (MAD) over a rolling window of N periods.

    Args:
        data (pd.Series): Input pandas Series with numerical data.
        N (int): Period for rolling calculation (default: 14).

    Returns:
        pd.Series: Series containing MAD values.
    """
    return (
        data -
        data.rolling(
            window=N).median()).abs().rolling(
        window=N).median()


def calculate_z_score(data: pd.Series, N: int = 14) -> pd.Series:
    """
    Calculate the Z-Score over a rolling window of N periods.

    Args:
        data (pd.Series): Input pandas Series with numerical data.
        N (int): Period for rolling calculation (default: 14).

    Returns:
        pd.Series: Series containing Z-Score values.
    """
    return (data - data.rolling(window=N).mean()) / \
        data.rolling(window=N).std()


def calculate_acceleration(data: pd.Series) -> pd.Series:
    """
    Calculate the Acceleration Indicator (second difference) of a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data.

    Returns:
        pd.Series: Series containing acceleration values.
    """
    return data.diff().diff()


def calculate_nth_autocorrelation_series(
        data: pd.Series,
        window: int = 20,
        lag: int = 1) -> pd.Series:
    """
    Calculate the nth-order autocorrelation (lag-k) as a rolling time series.

    Args:
        data (pd.Series): Input time series data.
        window (int): Size of the rolling window (default: 20).
        lag (int): Lag order for autocorrelation (default: 1).

    Returns:
        pd.Series: Rolling autocorrelation series at lag `k`.
    """
    def lagk_autocorr(x: np.ndarray) -> float:
        if len(x) <= lag or np.std(x) == 0:
            return np.nan
        x_mean = x.mean()
        denominator = np.sum((x - x_mean) ** 2)
        return np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / \
            denominator if denominator != 0 else np.nan

    return data.rolling(window=window).apply(lagk_autocorr, raw=True)


def calculate_cusum_positive(
        data: pd.Series,
        target: float = None) -> pd.Series:
    """
    Calculate the positive CUSUM (detect upward shifts) of a time series.

    Args:
        data (pd.Series): Time series input.
        target (float, optional): Target mean. Defaults to data mean.

    Returns:
        pd.Series: Positive CUSUM values.
    """
    if target is None:
        target = data.mean()
    cusum_pos = [0]
    for i in range(1, len(data)):
        cusum_pos.append(max(0, cusum_pos[-1] + (data.iloc[i] - target)))
    return pd.Series(cusum_pos, index=data.index)


def calculate_cusum_negative(
        data: pd.Series,
        target: float = None) -> pd.Series:
    """
    Calculate the negative CUSUM (detect downward shifts) of a time series.

    Args:
        data (pd.Series): Time series input.
        target (float, optional): Target mean. Defaults to data mean.

    Returns:
        pd.Series: Negative CUSUM values.
    """
    if target is None:
        target = data.mean()
    cusum_neg = [0]
    for i in range(1, len(data)):
        cusum_neg.append(min(0, cusum_neg[-1] + (data.iloc[i] - target)))
    return pd.Series(cusum_neg, index=data.index)


def calculate_tsi(data: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """
    Calculate the True Strength Index (TSI) for a given price series.

    Args:
        data (pd.Series): Input price series (e.g., close prices).
        r (int): Long EMA period. Default is 25.
        s (int): Short EMA period. Default is 13.

    Returns:
        pd.Series: TSI values.
    """
    momentum = data - data.shift(1)
    return 100 * (momentum.ewm(span=r,
                               adjust=False).mean().ewm(span=s,
                                                        adjust=False).mean() / momentum.abs().ewm(span=r,
                                                                                                  adjust=False).mean().ewm(span=s,
                                                                                                                           adjust=False).mean())


def calculate_kst(data: pd.Series,
                  roc_periods: list = [10, 15, 20, 30, 50],
                  smoothing_periods: list = [5, 10, 15, 20, 25]) -> pd.Series:
    """
    Calculate the Know Sure Thing (KST) indicator.

    Args:
        data (pd.Series): Input price series (e.g., close prices).
        roc_periods (list): List of periods for calculating ROC.
        smoothing_periods (list): List of smoothing periods (EMA spans).

    Returns:
        pd.Series: KST values.
    """
    return sum([roc.ewm(span=smoothing_periods[i], adjust=False).mean() for i, roc in enumerate(
        [(data - data.shift(period)) / data.shift(period) * 100 for period in roc_periods])])


def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR).

    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        period (int): Averaging period, default 14

    Returns:
        pd.Series: ATR values
    """
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1).rolling(window=period, min_periods=1).mean()


def get_base_fractal_points(
        data_high: pd.Series,
        data_low: pd.Series,
        N: int = 25) -> pd.Series:
    result = pd.Series(index=data_high.index, dtype=float)
    for i in range(N, len(data_high)):
        if data_high[i] == data_high[i - N + 1:i + 1].max():
            result.iloc[i] = data_high[i]
        elif data_low[i] == data_low[i - N + 1:i + 1].min():
            result.iloc[i] = data_low[i]
    return result


def get_higher_fractal_points(
        data_high: pd.Series,
        data_low: pd.Series,
        N: int = 25) -> pd.Series:
    periodF = int(N ** ((1 + 5 ** 0.5) / 2))
    result = pd.Series(index=data_high.index, dtype=float)
    for i in range(periodF, len(data_high)):
        if data_high[i] == data_high[i - periodF + 1:i + 1].max():
            result.iloc[i] = data_high[i]
        elif data_low[i] == data_low[i - periodF + 1:i + 1].min():
            result.iloc[i] = data_low[i]
    return result


def adx_proxy_from_typical_price(data: pd.Series, N: int = 14) -> pd.Series:
    return 100 * data.diff().abs().rolling(window=N).mean() / \
        data.rolling(window=N).mean()


def calculate_adx(
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        N: int = 14) -> pd.Series:
    """
    Calculate the ADX (Average Directional Index) based on price data (high, low, close).

    Args:
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        N (int): Smoothing period, default is 14

    Returns:
        pd.Series: The ADX (Average Directional Index) values as a pandas Series
    """
    plus_dm = data_high.diff()
    minus_dm = data_low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    smoothed_tr = pd.concat([data_high - data_low,
                             (data_high - data_close.shift(1)).abs(),
                             (data_low - data_close.shift(1)).abs()],
                            axis=1).max(axis=1).rolling(window=N,
                                                        min_periods=1).sum()
    plus_di = 100 * plus_dm.rolling(window=N,
                                    min_periods=1).sum() / smoothed_tr
    minus_di = 100 * minus_dm.rolling(window=N,
                                      min_periods=1).sum() / smoothed_tr
    return (100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            ).rolling(window=N, min_periods=1).mean()


def get_mean_line(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling mean (middle line).
    """
    return data_close.rolling(window=window).mean()


def get_up_band1(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Upper Band 1: Mean + 1 * Std.
    """
    mean = get_mean_line(data_close, window)
    std = data_close.rolling(window=window).std()
    return mean + std


def get_lo_band1(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Lower Band 1: Mean - 1 * Std.
    """
    mean = get_mean_line(data_close, window)
    std = data_close.rolling(window=window).std()
    return mean - std


def get_up_band2(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Upper Band 2: Mean + 2 * Std.
    """
    mean = get_mean_line(data_close, window)
    std = data_close.rolling(window=window).std()
    return mean + 2 * std


def get_lo_band2(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Lower Band 2: Mean - 2 * Std.
    """
    mean = get_mean_line(data_close, window)
    std = data_close.rolling(window=window).std()
    return mean - 2 * std


def get_band_range(data_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Range: UpBand2 - LoBand2.
    """
    upband2 = get_up_band2(data_close, window)
    loband2 = get_lo_band2(data_close, window)
    return upband2 - loband2


def calculate_parabolic_sar_with_trend(
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        af_start: float = 0.01,
        af_increment: float = 0.001,
        af_max: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR (Stop and Reverse) with trend identification for pandas Series of high, low, and close data.

    Args:
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        data_close (pd.Series): Input pandas Series with close demand data
        af_start (float): Starting acceleration factor (default: 0.02)
        af_increment (float): Increment for acceleration factor (default: 0.02)
        af_max (float): Maximum acceleration factor (default: 0.2)

    Returns:
        pd.Series: Tuple containing Parabolic SAR values
    """
    # Initialize variables
    sar = pd.Series(index=data_high.index, dtype=float)
    trend = pd.Series(index=data_high.index, dtype=float)
    af = af_start
    ep = 0.0
    prior_sar = 0.0

    if len(data_high) < 2:
        return sar, trend
    sar.iloc[0] = data_low.iloc[0]
    trend.iloc[0] = 1  # Assume uptrend initially

    for i in range(1, len(data_high)):
        high = data_high.iloc[i]
        low = data_low.iloc[i]
        prior_high = data_high.iloc[i - 1]
        prior_low = data_low.iloc[i - 1]

        if i == 1:
            trend.iloc[i] = 1 if data_high.iloc[1] > data_high.iloc[0] else -1
            ep = high if trend.iloc[i] == 1 else low
            prior_sar = sar.iloc[0]
            sar.iloc[i] = prior_sar
            continue

        if trend.iloc[i - 1] == 1:
            current_sar = prior_sar + af * (ep - prior_sar)
            current_sar = min(current_sar, prior_low, data_low.iloc[i - 2])
            if low < current_sar:
                trend.iloc[i] = -1
                current_sar = ep
                ep = low
                af = af_start
            else:
                trend.iloc[i] = 1
                if high > ep:
                    ep = high
                    af = min(af + af_increment, af_max)

        else:
            current_sar = prior_sar - af * (prior_sar - ep)
            current_sar = max(current_sar, prior_high, data_high.iloc[i - 2])
            if high > current_sar:
                trend.iloc[i] = 1
                current_sar = ep
                ep = high
                af = af_start
            else:
                trend.iloc[i] = -1
                if low < ep:
                    ep = low
                    af = min(af + af_increment, af_max)

        sar.iloc[i] = current_sar
        prior_sar = current_sar

    return sar


def calculate_tenkan_sen(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 9) -> pd.Series:
    """
    Calculate Tenkan-Sen (Conversion Line) for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n (int): Period for Tenkan-Sen calculation (default: 9)

    Returns:
        pd.Series: Series containing Tenkan-Sen values
    """
    return (data_high.rolling(window=n).max() +
            data_low.rolling(window=n).min()) / 2


def calculate_kijun_sen(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 26) -> pd.Series:
    """
    Calculate Kijun-Sen (Base Line) for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n (int): Period for Kijun-Sen calculation (default: 26)

    Returns:
        pd.Series: Series containing Kijun-Sen values
    """
    return (data_high.rolling(window=n).max() +
            data_low.rolling(window=n).min()) / 2


def calculate_chikou_span(data_close: pd.Series, n: int = 26) -> pd.Series:
    """
    Calculate Chikou Span (Lagging Span) for pandas Series of close data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        n (int): Period to shift back for Chikou Span (default: 26)

    Returns:
        pd.Series: Series containing Chikou Span values
    """
    return data_close.shift(-n)


def calculate_keltner_middle(data_close: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Keltner Channels Middle Line (EMA) for pandas Series of close data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        n (int): Period for EMA calculation (default: 20)

    Returns:
        pd.Series: Series containing Keltner Channels Middle Line (EMA) values
    """
    # Calculate EMA for the middle line
    middle_line = data_close.ewm(span=n, adjust=False).mean()
    return middle_line


def calculate_keltner_upper(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        n_ema: int = 20,
        n_atr: int = 10,
        multiplier: float = 2.0) -> pd.Series:
    """
    Calculate Keltner Channels Upper Band for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n_ema (int): Period for EMA calculation (default: 20)
        n_atr (int): Period for ATR calculation (default: 10)
        multiplier (float): Multiplier for ATR (default: 2.0)

    Returns:
        pd.Series: Series containing Keltner Channels Upper Band values
    """
    # Calculate EMA
    ema = data_close.ewm(span=n_ema, adjust=False).mean()

    # Calculate True Range (TR)
    tr1 = data_high - data_low
    tr2 = abs(data_high - data_close.shift(1))
    tr3 = abs(data_low - data_close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR
    atr = true_range.rolling(window=n_atr).mean()

    # Calculate Upper Band: EMA + (multiplier * ATR)
    upper_band = ema + (multiplier * atr)

    return upper_band


def calculate_keltner_lower(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        n_ema: int = 20,
        n_atr: int = 10,
        multiplier: float = 2.0) -> pd.Series:
    """
    Calculate Keltner Channels Lower Band for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n_ema (int): Period for EMA calculation (default: 20)
        n_atr (int): Period for ATR calculation (default: 10)
        multiplier (float): Multiplier for ATR (default: 2.0)

    Returns:
        pd.Series: Series containing Keltner Channels Lower Band values
    """
    return data_close.ewm(span=n_ema,
                          adjust=False).mean() - (multiplier * pd.concat([data_high - data_low,
                                                                          abs(data_high - data_close.shift(1)),
                                                                          abs(data_low - data_close.shift(1))],
                                                                         axis=1).max(axis=1).rolling(window=n_atr).mean())


def calculate_kdj_k(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 9,
        k_smooth: int = 3) -> pd.Series:
    """
    Calculate KDJ %K Line for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n (int): Period for RSV calculation (default: 9)
        k_smooth (int): Period for smoothing %K (default: 3)

    Returns:
        pd.Series: Series containing %K values
    """
    lowest_low = data_low.rolling(window=n).min()
    highest_high = data_high.rolling(window=n).max()
    rsv = ((data_close - lowest_low) / (highest_high - lowest_low)
           ).where(highest_high != lowest_low, 0) * 100
    k = pd.Series(index=data_close.index, dtype=float)
    k.iloc[:n - 1] = 50
    for i in range(n - 1, len(data_close)):
        if i == n - 1:
            k.iloc[i] = (2 / 3) * 50 + (1 / 3) * rsv.iloc[i]
        else:
            k.iloc[i] = (2 / 3) * k.iloc[i - 1] + (1 / 3) * rsv.iloc[i]
    return k


def calculate_kdj_d(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 9,
        k_smooth: int = 3,
        d_smooth: int = 3) -> pd.Series:
    """
    Calculate KDJ %D Line for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n (int): Period for RSV calculation (default: 9)
        k_smooth (int): Period for smoothing %K (default: 3)
        d_smooth (int): Period for smoothing %D (default: 3)

    Returns:
        pd.Series: Series containing %D values
    """
    k = calculate_kdj_k(data_close, data_high, data_low, n, k_smooth)
    d = pd.Series(index=k.index, dtype=float)
    d.iloc[:d_smooth - 1] = 50
    for i in range(d_smooth - 1, len(k)):
        if i == d_smooth - 1:
            d.iloc[i] = (2 / 3) * 50 + (1 / 3) * k.iloc[i]
        else:
            d.iloc[i] = (2 / 3) * d.iloc[i - 1] + (1 / 3) * k.iloc[i]
    return d


def calculate_kdj_j(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 9,
        k_smooth: int = 3,
        d_smooth: int = 3) -> pd.Series:
    """
    Calculate KDJ %J Line for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close demand data
        data_high (pd.Series): Input pandas Series with high demand data
        data_low (pd.Series): Input pandas Series with low demand data
        n (int): Period for RSV calculation (default: 9)
        k_smooth (int): Period for smoothing %K (default: 3)
        d_smooth (int): Period for smoothing %D (default: 3)

    Returns:
        pd.Series: Series containing %J values
    """
    return 3 * calculate_kdj_k(data_close,
                               data_high,
                               data_low,
                               n,
                               k_smooth) - 2 * calculate_kdj_d(data_close,
                                                               data_high,
                                                               data_low,
                                                               n,
                                                               k_smooth,
                                                               d_smooth)


def calculate_mcclellan_oscillator(
        data_close: pd.Series,
        short_ema: int = 19,
        long_ema: int = 39) -> pd.Series:
    """
    Calculate a modified McClellan Oscillator for a pandas Series of price data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        short_ema (int): Period for short-term EMA (default: 19)
        long_ema (int): Period for long-term EMA (default: 39)

    Returns:
        pd.Series: Series containing modified McClellan Oscillator values
    """
    price_diff = data_close.diff()
    net_advances = price_diff.where(
        price_diff > 0, 0) + price_diff.where(price_diff < 0, 0)
    return net_advances.ewm(span=short_ema, adjust=False).mean(
    ) - net_advances.ewm(span=long_ema, adjust=False).mean()


def calculate_donchian_upper(data_high: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Donchian Channels Upper Band for pandas Series of high data.

    Args:
        data_high (pd.Series): Input pandas Series with high price data
        n (int): Period for highest high calculation (default: 20)

    Returns:
        pd.Series: Series containing Donchian Channels Upper Band values
    """
    return data_high.rolling(window=n).max()


def calculate_donchian_lower(data_low: pd.Series, n: int = 20) -> pd.Series:
    """
    Calculate Donchian Channels Lower Band for pandas Series of low data.

    Args:
        data_low (pd.Series): Input pandas Series with low price data
        n (int): Period for lowest low calculation (default: 20)

    Returns:
        pd.Series: Series containing Donchian Channels Lower Band values
    """
    return data_low.rolling(window=n).min()


def calculate_donchian_middle(
        data_high: pd.Series,
        data_low: pd.Series,
        n: int = 20) -> pd.Series:
    """
    Calculate Donchian Channels Middle Line for pandas Series of high and low data.

    Args:
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        n (int): Period for calculation (default: 20)

    Returns:
        pd.Series: Series containing Donchian Channels Middle Line values
    """
    return (data_high.rolling(window=n).max() +
            data_low.rolling(window=n).min()) / 2


def calculate_supertrend_line(
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        atr_period: int = 10,
        multiplier: float = 3.0) -> pd.Series:
    """
    Calculate SuperTrend line for pandas Series of high, low, and close data.

    Args:
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        atr_period (int): Period for ATR calculation (default: 10)
        multiplier (float): Multiplier for ATR (default: 3.0)

    Returns:
        pd.Series: Series containing SuperTrend line values
    """
    tr1 = data_high - data_low
    tr2 = abs(data_high - data_close.shift(1))
    tr3 = abs(data_low - data_close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()

    supertrend = pd.Series(index=data_close.index, dtype=float)
    upper_band = pd.Series(index=data_close.index, dtype=float)
    lower_band = pd.Series(index=data_close.index, dtype=float)

    supertrend.iloc[0] = data_close.iloc[0]
    upper_band.iloc[0] = data_close.iloc[0] + (multiplier * atr.iloc[0])
    lower_band.iloc[0] = data_close.iloc[0] - (multiplier * atr.iloc[0])

    for i in range(1, len(data_close)):
        basic_upper = (
            data_high.iloc[i] + data_low.iloc[i]) / 2 + (multiplier * atr.iloc[i])
        basic_lower = (
            data_high.iloc[i] + data_low.iloc[i]) / 2 - (multiplier * atr.iloc[i])

        upper_band.iloc[i] = basic_upper if basic_upper < upper_band.iloc[i -
                                                                          1] or data_close.iloc[i -
                                                                                                1] > upper_band.iloc[i -
                                                                                                                     1] else upper_band.iloc[i -
                                                                                                                                             1]
        lower_band.iloc[i] = basic_lower if basic_lower > lower_band.iloc[i -
                                                                          1] or data_close.iloc[i -
                                                                                                1] < lower_band.iloc[i -
                                                                                                                     1] else lower_band.iloc[i -
                                                                                                                                             1]

        if data_close.iloc[i] > upper_band.iloc[i - 1]:
            supertrend.iloc[i] = lower_band.iloc[i]
        elif data_close.iloc[i] < lower_band.iloc[i - 1]:
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            supertrend.iloc[i] = lower_band.iloc[i] if supertrend.iloc[i -
                                                                       1] == lower_band.iloc[i - 1] else upper_band.iloc[i]

    return supertrend


def calculate_supertrend_trend(
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        atr_period: int = 10,
        multiplier: float = 3.0) -> pd.Series:
    """
    Calculate SuperTrend trend direction for pandas Series of high, low, and close data.

    Args:
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        atr_period (int): Period for ATR calculation (default: 10)
        multiplier (float): Multiplier for ATR (default: 3.0)

    Returns:
        pd.Series: Series containing trend direction values (1 for uptrend, -1 for downtrend)
    """
    tr1 = data_high - data_low
    tr2 = abs(data_high - data_close.shift(1))
    tr3 = abs(data_low - data_close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()

    trend = pd.Series(index=data_close.index, dtype=float)
    upper_band = pd.Series(index=data_close.index, dtype=float)
    lower_band = pd.Series(index=data_close.index, dtype=float)

    trend.iloc[0] = 1
    upper_band.iloc[0] = data_close.iloc[0] + (multiplier * atr.iloc[0])
    lower_band.iloc[0] = data_close.iloc[0] - (multiplier * atr.iloc[0])

    for i in range(1, len(data_close)):
        basic_upper = (
            data_high.iloc[i] + data_low.iloc[i]) / 2 + (multiplier * atr.iloc[i])
        basic_lower = (
            data_high.iloc[i] + data_low.iloc[i]) / 2 - (multiplier * atr.iloc[i])

        upper_band.iloc[i] = basic_upper if basic_upper < upper_band.iloc[i -
                                                                          1] or data_close.iloc[i -
                                                                                                1] > upper_band.iloc[i -
                                                                                                                     1] else upper_band.iloc[i -
                                                                                                                                             1]
        lower_band.iloc[i] = basic_lower if basic_lower > lower_band.iloc[i -
                                                                          1] or data_close.iloc[i -
                                                                                                1] < lower_band.iloc[i -
                                                                                                                     1] else lower_band.iloc[i -
                                                                                                                                             1]

        if data_close.iloc[i] > upper_band.iloc[i - 1]:
            trend.iloc[i] = 1
        elif data_close.iloc[i] < lower_band.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend


def calculate_moses_indicator(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        ma1_period: int = 10,
        ma2_period: int = 20,
        ma3_period: int = 50,
        ma4_period: int = 100,
        shock_threshold: float = 5.0,
        crash_threshold: float = 10.0) -> pd.Series:
    """
    Calculate MOSES Indicator value for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        ma1_period (int): Period for first moving average (default: 10)
        ma2_period (int): Period for second moving average (default: 20)
        ma3_period (int): Period for third moving average (default: 50)
        ma4_period (int): Period for fourth moving average (default: 100)
        shock_threshold (float): Percentage price drop for shock event (default: 5.0)
        crash_threshold (float): Percentage price drop for catastrophic event (default: 10.0)

    Returns:
        pd.Series: Series containing MOSES Indicator values
    """
    ma1 = data_close.rolling(window=ma1_period).mean()
    ma2 = data_close.rolling(window=ma2_period).mean()
    ma3 = data_close.rolling(window=ma3_period).mean()
    ma4 = data_close.rolling(window=ma4_period).mean()
    weekly_pct_change = data_close.pct_change() * 100
    moses = pd.Series(0.0, index=data_close.index)
    for i in range(1, len(data_close)):
        if (ma1.iloc[i] > ma2.iloc[i] > ma3.iloc[i] > ma4.iloc[i] and
                weekly_pct_change.iloc[i] > 0):
            moses.iloc[i] = 1.0
        elif (ma1.iloc[i] < ma2.iloc[i] < ma3.iloc[i] < ma4.iloc[i] or
              weekly_pct_change.iloc[i] < -crash_threshold):
            moses.iloc[i] = -1.0
        elif (moses.iloc[i - 1] < 0 and data_close.iloc[i] > ma1.iloc[i]):
            moses.iloc[i] = 0.5
        elif weekly_pct_change.iloc[i] < -shock_threshold:
            moses.iloc[i] = -0.5
        else:
            moses.iloc[i] = moses.iloc[i - 1]
    return moses


def calculate_moses_signal(
        data_close: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        ma1_period: int = 10,
        ma2_period: int = 20,
        ma3_period: int = 50,
        ma4_period: int = 100,
        shock_threshold: float = 5.0,
        crash_threshold: float = 10.0) -> pd.Series:
    """
    Calculate MOSES Signal type for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        ma1_period (int): Period for first moving average (default: 10)
        ma2_period (int): Period for second moving average (default: 20)
        ma3_period (int): Period for third moving average (default: 50)
        ma4_period (int): Period for fourth moving average (default: 100)
        shock_threshold (float): Percentage price drop for shock event (default: 5.0)
        crash_threshold (float): Percentage price drop for catastrophic event (default: 10.0)

    Returns:
        pd.Series: Series containing MOSES Signal types (1: Bull, -1: Bear, 0.5: Recovery, -0.5: Shock, -2: Catastrophe)
    """
    ma1 = data_close.rolling(window=ma1_period).mean()
    ma2 = data_close.rolling(window=ma2_period).mean()
    ma3 = data_close.rolling(window=ma3_period).mean()
    ma4 = data_close.rolling(window=ma4_period).mean()
    weekly_pct_change = data_close.pct_change() * 100
    signal = pd.Series(0.0, index=data_close.index)
    for i in range(1, len(data_close)):
        if (ma1.iloc[i] > ma2.iloc[i] > ma3.iloc[i] > ma4.iloc[i] and
                weekly_pct_change.iloc[i] > 0):
            signal.iloc[i] = 1.0
        elif ma1.iloc[i] < ma2.iloc[i] < ma3.iloc[i] < ma4.iloc[i]:
            signal.iloc[i] = -1.0
        elif (signal.iloc[i - 1] < 0 and data_close.iloc[i] > ma1.iloc[i]):
            signal.iloc[i] = 0.5
        elif weekly_pct_change.iloc[i] < -shock_threshold:
            signal.iloc[i] = -0.5
        elif weekly_pct_change.iloc[i] < -crash_threshold:
            signal.iloc[i] = -2.0
        else:
            signal.iloc[i] = signal.iloc[i - 1]
    return signal


def calculate_iv_rank(
        data_close: pd.Series,
        lookback: int = 252,
        period: int = 30) -> pd.Series:
    """
    Calculate Implied Volatility Rank (IVR) for pandas Series of close price data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        lookback (int): Number of days for historical comparison (default: 252)
        period (int): Period for calculating volatility (default: 30)

    Returns:
        pd.Series: Series containing IV Rank values (0 to 100)
    """
    daily_returns = data_close.pct_change().dropna()
    iv_proxy = daily_returns.rolling(window=period).std() * np.sqrt(252) * 100
    iv_high = iv_proxy.rolling(window=lookback).max()
    iv_low = iv_proxy.rolling(window=lookback).min()
    iv_rank = ((iv_proxy - iv_low) / (iv_high - iv_low)
               ).where(iv_high != iv_low, 0) * 100
    return iv_rank


def calculate_iv_percentile(
        data_close: pd.Series,
        lookback: int = 252,
        period: int = 30) -> pd.Series:
    """
    Calculate Implied Volatility Percentile (IVP) for pandas Series of close price data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        lookback (int): Number of days for historical comparison (default: 252)
        period (int): Period for calculating volatility (default: 30)

    Returns:
        pd.Series: Series containing IV Percentile values (0 to 100)
    """
    daily_returns = data_close.pct_change().dropna()
    iv_proxy = daily_returns.rolling(window=period).std() * np.sqrt(252) * 100
    iv_percentile = pd.Series(index=iv_proxy.index, dtype=float)
    for i in range(lookback - 1, len(iv_proxy)):
        past_values = iv_proxy.iloc[max(0, i - lookback + 1):i + 1]
        iv_percentile.iloc[i] = (
            past_values < iv_proxy.iloc[i]).sum() / len(past_values) * 100
    return iv_percentile.fillna(0)


def calculate_ml_rsi(
        data_close: pd.Series,
        rsi_length: int = 14,
        use_smoothing: bool = True,
        smoothing_length: int = 3,
        ma_type: str = "ALMA",
        alma_sigma: int = 4,
        use_knn: bool = True,
        knn_neighbors: int = 5,
        knn_lookback: int = 100,
        knn_weight: float = 0.4,
        feature_count: int = 3,
        use_filter: bool = True,
        filter_method: str = "Kalman",
        filter_strength: float = 0.3) -> pd.Series:
    """
    Calculate Machine Learning-enhanced RSI for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        rsi_length (int): Period for RSI calculation (default: 14)
        use_smoothing (bool): Apply moving average to RSI (default: True)
        smoothing_length (int): Period for moving average (default: 3)
        ma_type (str): Moving average type (default: "ALMA")
        alma_sigma (int): Sigma for ALMA (default: 4)
        use_knn (bool): Enable KNN machine learning (default: True)
        knn_neighbors (int): Number of KNN neighbors (default: 5)
        knn_lookback (int): Historical period for KNN (default: 100)
        knn_weight (float): Weight of KNN vs standard RSI (default: 0.4)
        feature_count (int): Number of features for KNN (default: 3)
        use_filter (bool): Apply additional smoothing (default: True)
        filter_method (str): Smoothing method (default: "Kalman")
        filter_strength (float): Smoothing parameter (default: 0.3)

    Returns:
        pd.Series: Series containing final RSI values
    """
    def calc_moving_average(src, length, ma_type, sigma):
        if ma_type == "SMA":
            return src.rolling(window=length).mean()
        elif ma_type == "EMA":
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == "DEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            return 2 * e1 - e2
        elif ma_type == "TEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            e3 = e2.ewm(span=length, adjust=False).mean()
            return 3 * (e1 - e2) + e3
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    weights) /
                weights.sum(),
                raw=True)
        elif ma_type == "SMMA":
            return src.ewm(alpha=1 / length, adjust=False).mean()
        elif ma_type == "HMA":
            wma1 = src.rolling(
                window=length //
                2).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length //
                        2 +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length //
                        2 +
                        1)),
                raw=True)
            wma2 = src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length +
                        1)),
                raw=True)
            diff = 2 * wma1 - wma2
            return diff.rolling(window=int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(
                1, int(np.sqrt(length)) + 1)) / np.sum(np.arange(1, int(np.sqrt(length)) + 1)), raw=True)
        elif ma_type == "LSMA":
            return src.rolling(window=length).apply(lambda x: np.polyfit(np.arange(length), x, 1)[
                1] + np.polyfit(np.arange(length), x, 1)[0] * (length - 1), raw=True)
        elif ma_type == "ALMA":
            m = np.floor(0.85 * (length - 1))
            s = length / sigma
            w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
            w = w / w.sum()
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    w),
                raw=True)
        return src

    def get_momentum(src, length):
        return src - src.shift(length)

    def get_volatility(src, length):
        return src.rolling(window=length).std()

    def get_slope(src, length):
        return src.rolling(
            window=length).apply(
            lambda x: np.polyfit(
                np.arange(length),
                x,
                1)[0],
            raw=True)

    def normalize(src, length):
        max_val = src.rolling(window=length).max()
        min_val = src.rolling(window=length).min()
        value_range = max_val - min_val
        return (src - min_val) / value_range.where(value_range != 0, 1)

    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    delta = data_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
    base_rsi = 100 - (100 / (1 + rs))
    smoothed_rsi = calc_moving_average(
        base_rsi,
        smoothing_length,
        ma_type,
        alma_sigma) if use_smoothing else base_rsi
    standard_rsi = smoothed_rsi

    enhanced_rsi = standard_rsi.copy()
    if use_knn:
        for i in range(knn_lookback, len(data_close)):
            current_features = []
            current_features.append(
                normalize(
                    standard_rsi,
                    knn_lookback).iloc[i])
            if feature_count >= 2:
                current_features.append(
                    normalize(
                        get_momentum(
                            standard_rsi,
                            3),
                        knn_lookback).iloc[i])
            if feature_count >= 3:
                current_features.append(
                    normalize(
                        get_volatility(
                            standard_rsi,
                            10),
                        knn_lookback).iloc[i])
            if feature_count >= 4:
                current_features.append(
                    normalize(
                        get_slope(
                            standard_rsi,
                            5),
                        knn_lookback).iloc[i])
            if feature_count >= 5:
                current_features.append(
                    normalize(
                        get_momentum(
                            data_close,
                            5),
                        knn_lookback).iloc[i])
            current_features = np.array(current_features)

            distances = []
            indices = []
            for j in range(1, knn_lookback + 1):
                if i - j >= 0:
                    historical_features = []
                    historical_features.append(
                        normalize(standard_rsi, knn_lookback).iloc[i - j])
                    if feature_count >= 2:
                        historical_features.append(normalize(get_momentum(
                            standard_rsi, 3), knn_lookback).iloc[i - j])
                    if feature_count >= 3:
                        historical_features.append(normalize(get_volatility(
                            standard_rsi, 10), knn_lookback).iloc[i - j])
                    if feature_count >= 4:
                        historical_features.append(normalize(get_slope(
                            standard_rsi, 5), knn_lookback).iloc[i - j])
                    if feature_count >= 5:
                        historical_features.append(normalize(
                            get_momentum(data_close, 5), knn_lookback).iloc[i - j])
                    historical_features = np.array(historical_features)
                    if len(historical_features) == len(current_features):
                        dist = euclidean_distance(
                            current_features, historical_features)
                        distances.append(dist)
                        indices.append(i - j)

            if distances:
                sorted_pairs = sorted(zip(distances, indices))
                distances, indices = zip(*sorted_pairs)
                weighted_sum = 0.0
                weight_sum = 0.0
                effective_k = min(knn_neighbors, len(distances))
                for k in range(effective_k):
                    idx = indices[k]
                    dist = distances[k]
                    weight = 1.0 / dist if dist >= 0.0001 else 1.0
                    neighbor_rsi = standard_rsi.iloc[idx]
                    if not np.isnan(neighbor_rsi):
                        weighted_sum += neighbor_rsi * weight
                        weight_sum += weight
                if weight_sum > 0:
                    knn_rsi = weighted_sum / weight_sum
                    enhanced_rsi.iloc[i] = (
                        1.0 - knn_weight) * standard_rsi.iloc[i] + knn_weight * knn_rsi
                    enhanced_rsi.iloc[i] = max(
                        0.0, min(100.0, enhanced_rsi.iloc[i]))

    final_rsi = enhanced_rsi
    if use_filter:
        if filter_method == "Kalman":
            kf_rsi = final_rsi.copy()
            alpha = filter_strength
            for i in range(1, len(kf_rsi)):
                kf_rsi.iloc[i] = (1 - alpha) * kf_rsi.iloc[i - 1] + alpha * \
                    final_rsi.iloc[i] if not np.isnan(kf_rsi.iloc[i - 1]) else final_rsi.iloc[i]
            final_rsi = kf_rsi
        elif filter_method == "DoubleEMA":
            ema1 = final_rsi.ewm(
                span=round(
                    filter_strength * 10),
                adjust=False).mean()
            final_rsi = ema1.ewm(
                span=round(
                    filter_strength * 5),
                adjust=False).mean()
        elif filter_method == "ALMA":
            m = np.floor(0.85 * (round(filter_strength * 20) - 1))
            s = round(filter_strength * 20) / 6
            w = np.exp(-((np.arange(round(filter_strength * 20)) - m)
                       ** 2) / (2 * s ** 2))
            w = w / w.sum()
            final_rsi = final_rsi.rolling(
                window=round(
                    filter_strength *
                    20)).apply(
                lambda x: np.dot(
                    x,
                    w) if len(x) == len(w) else np.nan,
                raw=True)

    return final_rsi


def calculate_ml_rsi_overbought(
        data_close: pd.Series,
        rsi_length: int = 14,
        use_smoothing: bool = True,
        smoothing_length: int = 3,
        ma_type: str = "ALMA",
        alma_sigma: int = 4,
        use_knn: bool = True,
        knn_lookback: int = 100,
        feature_count: int = 3) -> pd.Series:
    """
    Calculate Machine Learning-enhanced RSI Overbought Threshold for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        rsi_length (int): Period for RSI calculation (default: 14)
        use_smoothing (bool): Apply moving average to RSI (default: True)
        smoothing_length (int): Period for moving average (default: 3)
        ma_type (str): Moving average type (default: "ALMA")
        alma_sigma (int): Sigma for ALMA (default: 4)
        use_knn (bool): Enable KNN machine learning (default: True)
        knn_neighbors (int): Number of KNN neighbors (default: 5)
        knn_lookback (int): Historical period for KNN (default: 100)
        knn_weight (float): Weight of KNN vs standard RSI (default: 0.4)
        feature_count (int): Number of features for KNN (default: 3)

    Returns:
        pd.Series: Series containing overbought threshold values
    """
    def calc_moving_average(src, length, ma_type, sigma):
        if ma_type == "SMA":
            return src.rolling(window=length).mean()
        elif ma_type == "EMA":
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == "DEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            return 2 * e1 - e2
        elif ma_type == "TEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            e3 = e2.ewm(span=length, adjust=False).mean()
            return 3 * (e1 - e2) + e3
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    weights) /
                weights.sum(),
                raw=True)
        elif ma_type == "SMMA":
            return src.ewm(alpha=1 / length, adjust=False).mean()
        elif ma_type == "HMA":
            wma1 = src.rolling(
                window=length //
                2).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length //
                        2 +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length //
                        2 +
                        1)),
                raw=True)
            wma2 = src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length +
                        1)),
                raw=True)
            diff = 2 * wma1 - wma2
            return diff.rolling(window=int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(
                1, int(np.sqrt(length)) + 1)) / np.sum(np.arange(1, int(np.sqrt(length)) + 1)), raw=True)
        elif ma_type == "LSMA":
            return src.rolling(window=length).apply(lambda x: np.polyfit(np.arange(length), x, 1)[
                1] + np.polyfit(np.arange(length), x, 1)[0] * (length - 1), raw=True)
        elif ma_type == "ALMA":
            m = np.floor(0.85 * (length - 1))
            s = length / sigma
            w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
            w = w / w.sum()
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    w),
                raw=True)
        return src

    def get_momentum(src, length):
        return src - src.shift(length)

    def get_volatility(src, length):
        return src.rolling(window=length).std()

    def get_slope(src, length):
        return src.rolling(
            window=length).apply(
            lambda x: np.polyfit(
                np.arange(length),
                x,
                1)[0],
            raw=True)

    def normalize(src, length):
        max_val = src.rolling(window=length).max()
        min_val = src.rolling(window=length).min()
        value_range = max_val - min_val
        return (src - min_val) / value_range.where(value_range != 0, 1)

    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    delta = data_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
    base_rsi = 100 - (100 / (1 + rs))
    smoothed_rsi = calc_moving_average(
        base_rsi,
        smoothing_length,
        ma_type,
        alma_sigma) if use_smoothing else base_rsi
    standard_rsi = smoothed_rsi

    overbought = pd.Series(70.0, index=data_close.index)
    if use_knn:
        for i in range(knn_lookback, len(data_close)):
            current_features = []
            current_features.append(
                normalize(
                    standard_rsi,
                    knn_lookback).iloc[i])
            if feature_count >= 2:
                current_features.append(
                    normalize(
                        get_momentum(
                            standard_rsi,
                            3),
                        knn_lookback).iloc[i])
            if feature_count >= 3:
                current_features.append(
                    normalize(
                        get_volatility(
                            standard_rsi,
                            10),
                        knn_lookback).iloc[i])
            if feature_count >= 4:
                current_features.append(
                    normalize(
                        get_slope(
                            standard_rsi,
                            5),
                        knn_lookback).iloc[i])
            if feature_count >= 5:
                current_features.append(
                    normalize(
                        get_momentum(
                            data_close,
                            5),
                        knn_lookback).iloc[i])
            current_features = np.array(current_features)

            distances = []
            indices = []
            overbought_candidates = []
            for j in range(1, knn_lookback + 1):
                if i - j >= 0:
                    historical_features = []
                    historical_features.append(
                        normalize(standard_rsi, knn_lookback).iloc[i - j])
                    if feature_count >= 2:
                        historical_features.append(normalize(get_momentum(
                            standard_rsi, 3), knn_lookback).iloc[i - j])
                    if feature_count >= 3:
                        historical_features.append(normalize(get_volatility(
                            standard_rsi, 10), knn_lookback).iloc[i - j])
                    if feature_count >= 4:
                        historical_features.append(normalize(get_slope(
                            standard_rsi, 5), knn_lookback).iloc[i - j])
                    if feature_count >= 5:
                        historical_features.append(normalize(
                            get_momentum(data_close, 5), knn_lookback).iloc[i - j])
                    historical_features = np.array(historical_features)
                    if len(historical_features) == len(current_features):
                        dist = euclidean_distance(
                            current_features, historical_features)
                        distances.append(dist)
                        indices.append(i - j)
                        if i - j + 5 < len(data_close) and not np.isnan(
                                data_close.iloc[i - j + 5]) and not np.isnan(data_close.iloc[i - j]):
                            future_return = data_close.iloc[i - \
                                j + 5] - data_close.iloc[i - j]
                            if future_return > data_close.iloc[i - j] * 0.02:
                                overbought_candidates.append(
                                    standard_rsi.iloc[i - j])

            if overbought_candidates:
                overbought.iloc[i] = np.mean(overbought_candidates)

    return overbought


def calculate_ml_rsi_oversold(
        data_close: pd.Series,
        rsi_length: int = 14,
        use_smoothing: bool = True,
        smoothing_length: int = 3,
        ma_type: str = "ALMA",
        alma_sigma: int = 4,
        use_knn: bool = True,
        knn_lookback: int = 100,
        feature_count: int = 3) -> pd.Series:
    """
    Calculate Machine Learning-enhanced RSI Oversold Threshold for pandas Series of close, high, and low data.

    Args:
        data_close (pd.Series): Input pandas Series with close price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        rsi_length (int): Period for RSI calculation (default: 14)
        use_smoothing (bool): Apply moving average to RSI (default: True)
        smoothing_length (int): Period for moving average (default: 3)
        ma_type (str): Moving average type (default: "ALMA")
        alma_sigma (int): Sigma for ALMA (default: 4)
        use_knn (bool): Enable KNN machine learning (default: True)
        knn_neighbors (int): Number of KNN neighbors (default: 5)
        knn_lookback (int): Historical period for KNN (default: 100)
        knn_weight (float): Weight of KNN vs standard RSI (default: 0.4)
        feature_count (int): Number of features for KNN (default: 3)

    Returns:
        pd.Series: Series containing oversold threshold values
    """
    def calc_moving_average(src, length, ma_type, sigma):
        if ma_type == "SMA":
            return src.rolling(window=length).mean()
        elif ma_type == "EMA":
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == "DEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            return 2 * e1 - e2
        elif ma_type == "TEMA":
            e1 = src.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            e3 = e2.ewm(span=length, adjust=False).mean()
            return 3 * (e1 - e2) + e3
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    weights) /
                weights.sum(),
                raw=True)
        elif ma_type == "SMMA":
            return src.ewm(alpha=1 / length, adjust=False).mean()
        elif ma_type == "HMA":
            wma1 = src.rolling(
                window=length //
                2).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length //
                        2 +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length //
                        2 +
                        1)),
                raw=True)
            wma2 = src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    np.arange(
                        1,
                        length +
                        1)) /
                np.sum(
                    np.arange(
                        1,
                        length +
                        1)),
                raw=True)
            diff = 2 * wma1 - wma2
            return diff.rolling(window=int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(
                1, int(np.sqrt(length)) + 1)) / np.sum(np.arange(1, int(np.sqrt(length)) + 1)), raw=True)
        elif ma_type == "LSMA":
            return src.rolling(window=length).apply(lambda x: np.polyfit(np.arange(length), x, 1)[
                1] + np.polyfit(np.arange(length), x, 1)[0] * (length - 1), raw=True)
        elif ma_type == "ALMA":
            m = np.floor(0.85 * (length - 1))
            s = length / sigma
            w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
            w = w / w.sum()
            return src.rolling(
                window=length).apply(
                lambda x: np.dot(
                    x,
                    w),
                raw=True)
        return src

    def get_momentum(src, length):
        return src - src.shift(length)

    def get_volatility(src, length):
        return src.rolling(window=length).std()

    def get_slope(src, length):
        return src.rolling(
            window=length).apply(
            lambda x: np.polyfit(
                np.arange(length),
                x,
                1)[0],
            raw=True)

    def normalize(src, length):
        max_val = src.rolling(window=length).max()
        min_val = src.rolling(window=length).min()
        value_range = max_val - min_val
        return (src - min_val) / value_range.where(value_range != 0, 1)

    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    delta = data_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
    base_rsi = 100 - (100 / (1 + rs))
    smoothed_rsi = calc_moving_average(
        base_rsi,
        smoothing_length,
        ma_type,
        alma_sigma) if use_smoothing else base_rsi
    standard_rsi = smoothed_rsi

    oversold = pd.Series(30.0, index=data_close.index)
    if use_knn:
        for i in range(knn_lookback, len(data_close)):
            current_features = []
            current_features.append(
                normalize(
                    standard_rsi,
                    knn_lookback).iloc[i])
            if feature_count >= 2:
                current_features.append(
                    normalize(
                        get_momentum(
                            standard_rsi,
                            3),
                        knn_lookback).iloc[i])
            if feature_count >= 3:
                current_features.append(
                    normalize(
                        get_volatility(
                            standard_rsi,
                            10),
                        knn_lookback).iloc[i])
            if feature_count >= 4:
                current_features.append(
                    normalize(
                        get_slope(
                            standard_rsi,
                            5),
                        knn_lookback).iloc[i])
            if feature_count >= 5:
                current_features.append(
                    normalize(
                        get_momentum(
                            data_close,
                            5),
                        knn_lookback).iloc[i])
            current_features = np.array(current_features)

            distances = []
            indices = []
            oversold_candidates = []
            for j in range(1, knn_lookback + 1):
                if i - j >= 0:
                    historical_features = []
                    historical_features.append(
                        normalize(standard_rsi, knn_lookback).iloc[i - j])
                    if feature_count >= 2:
                        historical_features.append(normalize(get_momentum(
                            standard_rsi, 3), knn_lookback).iloc[i - j])
                    if feature_count >= 3:
                        historical_features.append(normalize(get_volatility(
                            standard_rsi, 10), knn_lookback).iloc[i - j])
                    if feature_count >= 4:
                        historical_features.append(normalize(get_slope(
                            standard_rsi, 5), knn_lookback).iloc[i - j])
                    if feature_count >= 5:
                        historical_features.append(normalize(
                            get_momentum(data_close, 5), knn_lookback).iloc[i - j])
                    historical_features = np.array(historical_features)
                    if len(historical_features) == len(current_features):
                        dist = euclidean_distance(
                            current_features, historical_features)
                        distances.append(dist)
                        indices.append(i - j)
                        if i - j + 5 < len(data_close) and not np.isnan(
                                data_close.iloc[i - j + 5]) and not np.isnan(data_close.iloc[i - j]):
                            future_return = data_close.iloc[i - \
                                j + 5] - data_close.iloc[i - j]
                            if future_return < data_close.iloc[i - j] * -0.02:
                                oversold_candidates.append(
                                    standard_rsi.iloc[i - j])

            if oversold_candidates:
                oversold.iloc[i] = np.mean(oversold_candidates)

    return oversold


def calculate_zigzag(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        deviation: float = 5.0,
        depth: int = 10) -> pd.Series:
    """
    Calculate the ZigZag indicator as a pd.Series for time series prediction, identifying significant price reversals.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        deviation (float): Minimum percentage price change to form a new pivot (default: 5.0)
        depth (int): Number of bars to confirm a pivot (minimum 2, default: 10)

    Returns:
        pd.Series: Series with pivot prices at reversal points, 0 elsewhere, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if deviation <= 0 or deviation > 100:
        raise ValueError("Deviation must be between 0.00001 and 100")

    if depth < 2:
        raise ValueError("Depth must be at least 2")

    # Initialize output series with zeros
    zigzag = pd.Series(0.0, index=data_close.index)
    deviation = deviation / 100  # Convert percentage to decimal
    last_pivot_price = data_close.iloc[0]
    last_pivot_index = data_close.index[0]
    last_pivot_type = None
    direction = 0  # 0: undefined, 1: up, -1: down

    def is_pivot_high(index, depth):
        """Check if the bar at index is a pivot high."""
        idx = data_high.index.get_loc(index)
        if idx < depth or idx >= len(data_high) - depth:
            return False
        center = data_high.iloc[idx]
        for i in range(idx - depth, idx + depth + 1):
            if i != idx and data_high.iloc[i] > center:
                return False
        return True

    def is_pivot_low(index, depth):
        """Check if the bar at index is a pivot low."""
        idx = data_low.index.get_loc(index)
        if idx < depth or idx >= len(data_low) - depth:
            return False
        center = data_low.iloc[idx]
        for i in range(idx - depth, idx + depth + 1):
            if i != idx and data_low.iloc[i] < center:
                return False
        return True

    # Iterate through the data to find pivots
    for i in range(depth, len(data_close)):
        current_index = data_close.index[i]
        current_high = data_high.iloc[i]
        current_low = data_low.iloc[i]

        # Check for pivot high
        if is_pivot_high(current_index, depth // 2):
            price_change = (
                current_high /
                last_pivot_price -
                1) if last_pivot_price > 0 else 0
            if (direction <= 0 and price_change >= deviation) or (
                    direction == 1 and current_high > last_pivot_price):
                if last_pivot_type != 'high' or price_change >= deviation:
                    if last_pivot_type == 'low':
                        zigzag.loc[last_pivot_index] = last_pivot_price
                    zigzag.loc[current_index] = current_high
                    last_pivot_price = current_high
                    last_pivot_index = current_index
                    last_pivot_type = 'high'
                    direction = 1

        # Check for pivot low
        if is_pivot_low(current_index, depth // 2):
            price_change = (
                last_pivot_price /
                current_low -
                1) if current_low > 0 else 0
            if (direction >= 0 and price_change >= deviation) or (
                    direction == -1 and current_low < last_pivot_price):
                if last_pivot_type != 'low' or price_change >= deviation:
                    if last_pivot_type == 'high':
                        zigzag.loc[last_pivot_index] = last_pivot_price
                    zigzag.loc[current_index] = current_low
                    last_pivot_price = current_low
                    last_pivot_index = current_index
                    last_pivot_type = 'low'
                    direction = -1

    # Clean up consecutive pivots of the same type
    temp_zigzag = zigzag[zigzag != 0]
    if len(temp_zigzag) > 1:
        cleaned_zigzag = pd.Series(0.0, index=data_close.index)
        i = 0
        while i < len(temp_zigzag):
            current_index = temp_zigzag.index[i]
            current_price = temp_zigzag.iloc[i]
            cleaned_zigzag.loc[current_index] = current_price
            j = i + 1
            current_type = 'high' if (
                i == 0 or temp_zigzag.iloc[i] > temp_zigzag.iloc[i - 1]) else 'low'
            while j < len(temp_zigzag) and ((current_type == 'high' and temp_zigzag.iloc[j] > temp_zigzag.iloc[j - 1]) or (
                    current_type == 'low' and temp_zigzag.iloc[j] < temp_zigzag.iloc[j - 1])):
                if (current_type == 'high' and temp_zigzag.iloc[j] > current_price) or (
                        current_type == 'low' and temp_zigzag.iloc[j] < current_price):
                    cleaned_zigzag.loc[current_index] = 0.0
                    current_index = temp_zigzag.index[j]
                    current_price = temp_zigzag.iloc[j]
                    cleaned_zigzag.loc[current_index] = current_price
                j += 1
            i = j
        zigzag = cleaned_zigzag

    return zigzag


def calculate_cci_turbo(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        cci_turbo_length: int = 6) -> pd.Series:
    """
    Calculate Woodies CCI Turbo indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        cci_turbo_length (int): Length for Turbo CCI (default: 6, min: 3, max: 14)

    Returns:
        pd.Series: Series containing Turbo CCI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if not (3 <= cci_turbo_length <= 14):
        raise ValueError("cci_turbo_length must be between 3 and 14")

    typical_price = (data_high + data_low + data_close) / 3
    sma_tp = typical_price.rolling(window=cci_turbo_length).mean()
    mean_dev = typical_price.rolling(
        window=cci_turbo_length).apply(
        lambda x: np.mean(
            np.abs(
                x -
                x.mean())),
        raw=True)
    cci_turbo = (typical_price - sma_tp) / (0.015 * mean_dev)

    return cci_turbo


def calculate_cci_14(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        cci_14_length: int = 14) -> pd.Series:
    """
    Calculate Woodies CCI 14-period indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        cci_14_length (int): Length for 14-period CCI (default: 14, min: 7, max: 20)

    Returns:
        pd.Series: Series containing 14-period CCI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if not (7 <= cci_14_length <= 20):
        raise ValueError("cci_14_length must be between 7 and 20")
    typical_price = (data_high + data_low + data_close) / 3
    sma_tp = typical_price.rolling(window=cci_14_length).mean()
    mean_dev = typical_price.rolling(
        window=cci_14_length).apply(
        lambda x: np.mean(
            np.abs(
                x -
                x.mean())),
        raw=True)
    cci_14 = (typical_price - sma_tp) / (0.015 * mean_dev)

    return cci_14


def calculate_williams_percent_r(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 14) -> pd.Series:
    """
    Calculate Williams Percent Range (%R) indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for %R calculation (default: 14)

    Returns:
        pd.Series: Series containing Williams %R values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")
    max_high = data_high.rolling(window=length).max()
    min_low = data_low.rolling(window=length).min()
    percent_r = 100 * (data_close - max_high) / (max_high -
                                                 min_low).where(max_high != min_low, np.nan)
    return percent_r


def calculate_williams_percent_r_signal(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 14) -> pd.Series:
    """
    Calculate Williams Percent Range (%R) signal as a pd.Series, indicating overbought, oversold, or neutral conditions.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for %R calculation (default: 14)

    Returns:
        pd.Series: Series with signal values (+1 for %R > -20, -1 for %R < -80, 0 for -80 <= %R <= -20), aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")
    max_high = data_high.rolling(window=length).max()
    min_low = data_low.rolling(window=length).min()
    percent_r = 100 * (data_close - max_high) / (max_high -
                                                 min_low).where(max_high != min_low, np.nan)
    signal = pd.Series(0, index=data_close.index, dtype=int)
    signal[percent_r > -20] = 1
    signal[percent_r < -80] = -1
    return signal


def calculate_alligator_jaw(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        jaw_length: int = 13,
        jaw_offset: int = 8) -> pd.Series:
    """
    Calculate Williams Alligator Jaw line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        jaw_length (int): Length for Jaw SMMA (default: 13, min: 1)
        jaw_offset (int): Offset for Jaw line (default: 8)

    Returns:
        pd.Series: Series containing Jaw SMMA values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if jaw_length < 1:
        raise ValueError("Jaw length must be at least 1")

    def smma(src, length):
        """Calculate Smoothed Moving Average (SMMA)."""
        smma = pd.Series(np.nan, index=src.index)
        sma_initial = src.rolling(window=length).mean()
        smma.iloc[length - 1] = sma_initial.iloc[length - 1]

        for i in range(length, len(src)):
            if not np.isnan(smma.iloc[i - 1]):
                smma.iloc[i] = (smma.iloc[i - 1] *
                                (length - 1) + src.iloc[i]) / length

        return smma
    hl2 = (data_high + data_low) / 2
    jaw = smma(hl2, jaw_length).shift(jaw_offset)
    return jaw


def calculate_alligator_teeth(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        teeth_length: int = 8,
        teeth_offset: int = 5) -> pd.Series:
    """
    Calculate Williams Alligator Teeth line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        teeth_length (int): Length for Teeth SMMA (default: 8, min: 1)
        teeth_offset (int): Offset for Teeth line (default: 5)

    Returns:
        pd.Series: Series containing Teeth SMMA values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if teeth_length < 1:
        raise ValueError("Teeth length must be at least 1")

    def smma(src, length):
        """Calculate Smoothed Moving Average (SMMA)."""
        smma = pd.Series(np.nan, index=src.index)
        sma_initial = src.rolling(window=length).mean()
        smma.iloc[length - 1] = sma_initial.iloc[length - 1]

        for i in range(length, len(src)):
            if not np.isnan(smma.iloc[i - 1]):
                smma.iloc[i] = (smma.iloc[i - 1] *
                                (length - 1) + src.iloc[i]) / length

        return smma
    hl2 = (data_high + data_low) / 2
    teeth = smma(hl2, teeth_length).shift(teeth_offset)
    return teeth


def calculate_alligator_lips(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        lips_length: int = 5,
        lips_offset: int = 3) -> pd.Series:
    """
    Calculate Williams Alligator Lips line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        lips_length (int): Length for Lips SMMA (default: 5, min: 1)
        lips_offset (int): Offset for Lips line (default: 3)

    Returns:
        pd.Series: Series containing Lips SMMA values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if lips_length < 1:
        raise ValueError("Lips length must be at least 1")

    def smma(src, length):
        """Calculate Smoothed Moving Average (SMMA)."""
        smma = pd.Series(np.nan, index=src.index)
        sma_initial = src.rolling(window=length).mean()
        smma.iloc[length - 1] = sma_initial.iloc[length - 1]

        for i in range(length, len(src)):
            if not np.isnan(smma.iloc[i - 1]):
                smma.iloc[i] = (smma.iloc[i - 1] *
                                (length - 1) + src.iloc[i]) / length

        return smma
    hl2 = (data_high + data_low) / 2
    lips = smma(hl2, lips_length).shift(lips_offset)
    return lips


def calculate_vortex_positive(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Calculate Vortex Indicator Positive (VI+) line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        period (int): Lookback period for VI+ calculation (default: 14, min: 2)

    Returns:
        pd.Series: Series containing VI+ values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if period < 2:
        raise ValueError("Period must be at least 2")
    vmp = np.abs(data_high - data_low.shift(1))
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    vmp_sum = vmp.rolling(window=period).sum()
    str_sum = tr.rolling(window=period).sum()
    vip = vmp_sum / str_sum.where(str_sum != 0, np.nan)
    return vip


def calculate_vortex_negative(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Calculate Vortex Indicator Negative (VI-) line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        period (int): Lookback period for VI- calculation (default: 14, min: 2)

    Returns:
        pd.Series: Series containing VI- values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if period < 2:
        raise ValueError("Period must be at least 2")
    vmm = np.abs(data_low - data_high.shift(1))
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    vmm_sum = vmm.rolling(window=period).sum()
    str_sum = tr.rolling(window=period).sum()
    vim = vmm_sum / str_sum.where(str_sum != 0, np.nan)
    return vim


def calculate_volatility_stop_vstop(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 20,
        factor: float = 2.0) -> pd.Series:
    """
    Calculate Volatility Stop (vstop) line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for ATR calculation (default: 20, min: 2)
        factor (float): Multiplier for ATR (default: 2.0, min: 0.25)

    Returns:
        pd.Series: Series containing Volatility Stop prices, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 2:
        raise ValueError("Length must be at least 2")
    if factor < 0.25:
        raise ValueError("Factor must be at least 0.25")
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    src = data_close
    max_price = pd.Series(np.nan, index=src.index)
    min_price = pd.Series(np.nan, index=src.index)
    uptrend = pd.Series(False, index=src.index)
    vstop = pd.Series(np.nan, index=src.index)
    max_price.iloc[0] = src.iloc[0]
    min_price.iloc[0] = src.iloc[0]
    uptrend.iloc[0] = True
    vstop.iloc[0] = src.iloc[0]
    for i in range(1, len(src)):
        atr_m = atr.iloc[i] * \
            factor if not np.isnan(atr.iloc[i]) else tr.iloc[i]
        max_price.iloc[i] = max(max_price.iloc[i -
                                               1], src.iloc[i]) if not np.isnan(max_price.iloc[i -
                                                                                               1]) else src.iloc[i]
        min_price.iloc[i] = min(min_price.iloc[i -
                                               1], src.iloc[i]) if not np.isnan(min_price.iloc[i -
                                                                                               1]) else src.iloc[i]
        if uptrend.iloc[i - 1]:
            vstop.iloc[i] = max(vstop.iloc[i -
                                           1], max_price.iloc[i] -
                                atr_m) if not np.isnan(vstop.iloc[i -
                                                                  1]) else src.iloc[i]
        else:
            vstop.iloc[i] = min(vstop.iloc[i -
                                           1], min_price.iloc[i] +
                                atr_m) if not np.isnan(vstop.iloc[i -
                                                                  1]) else src.iloc[i]
        uptrend.iloc[i] = src.iloc[i] - vstop.iloc[i] >= 0
        if uptrend.iloc[i] != uptrend.iloc[i - 1]:
            max_price.iloc[i] = src.iloc[i]
            min_price.iloc[i] = src.iloc[i]
            vstop.iloc[i] = max_price.iloc[i] - \
                atr_m if uptrend.iloc[i] else min_price.iloc[i] + atr_m
    return vstop


def calculate_volatility_stop_uptrend(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 20,
        factor: float = 2.0) -> pd.Series:
    """
    Calculate Volatility Stop uptrend indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for ATR calculation (default: 20, min: 2)
        factor (float): Multiplier for ATR (default: 2.0, min: 0.25)

    Returns:
        pd.Series: Series containing boolean uptrend values (True for uptrend, False for downtrend), aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 2:
        raise ValueError("Length must be at least 2")
    if factor < 0.25:
        raise ValueError("Factor must be at least 0.25")
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    src = data_close
    max_price = pd.Series(np.nan, index=src.index)
    min_price = pd.Series(np.nan, index=src.index)
    uptrend = pd.Series(False, index=src.index)
    vstop = pd.Series(np.nan, index=src.index)
    max_price.iloc[0] = src.iloc[0]
    min_price.iloc[0] = src.iloc[0]
    uptrend.iloc[0] = True
    vstop.iloc[0] = src.iloc[0]
    for i in range(1, len(src)):
        atr_m = atr.iloc[i] * \
            factor if not np.isnan(atr.iloc[i]) else tr.iloc[i]
        max_price.iloc[i] = max(max_price.iloc[i -
                                               1], src.iloc[i]) if not np.isnan(max_price.iloc[i -
                                                                                               1]) else src.iloc[i]
        min_price.iloc[i] = min(min_price.iloc[i -
                                               1], src.iloc[i]) if not np.isnan(min_price.iloc[i -
                                                                                               1]) else src.iloc[i]
        if uptrend.iloc[i - 1]:
            vstop.iloc[i] = max(vstop.iloc[i -
                                           1], max_price.iloc[i] -
                                atr_m) if not np.isnan(vstop.iloc[i -
                                                                  1]) else src.iloc[i]
        else:
            vstop.iloc[i] = min(vstop.iloc[i -
                                           1], min_price.iloc[i] +
                                atr_m) if not np.isnan(vstop.iloc[i -
                                                                  1]) else src.iloc[i]
        uptrend.iloc[i] = src.iloc[i] - vstop.iloc[i] >= 0
        if uptrend.iloc[i] != uptrend.iloc[i - 1]:
            max_price.iloc[i] = src.iloc[i]
            min_price.iloc[i] = src.iloc[i]
            vstop.iloc[i] = max_price.iloc[i] - \
                atr_m if uptrend.iloc[i] else min_price.iloc[i] + atr_m
    return uptrend


def calculate_smi_ergodic_oscillator(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        longlen: int = 20,
        shortlen: int = 5,
        siglen: int = 5) -> pd.Series:
    """
    Calculate SMI Ergodic Oscillator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        longlen (int): Long length for TSI calculation (default: 20, min: 1)
        shortlen (int): Short length for TSI calculation (default: 5, min: 1)
        siglen (int): Signal line EMA length (default: 5, min: 1)

    Returns:
        pd.Series: Series containing SMI Ergodic Oscillator values (TSI - Signal), aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if longlen < 1 or shortlen < 1 or siglen < 1:
        raise ValueError("Length parameters must be at least 1")

    def calculate_tsi(src, shortlen, longlen):
        """Calculate True Strength Index (TSI)."""
        delta = src.diff()
        abs_delta = delta.abs()
        ema_short = delta.ewm(span=shortlen, adjust=False).mean()
        ema_long = ema_short.ewm(span=longlen, adjust=False).mean()
        abs_ema_short = abs_delta.ewm(span=shortlen, adjust=False).mean()
        abs_ema_long = abs_ema_short.ewm(span=longlen, adjust=False).mean()
        tsi = 100 * ema_long / abs_ema_long.where(abs_ema_long != 0, np.nan)
        return tsi
    erg = calculate_tsi(data_close, shortlen, longlen)
    sig = erg.ewm(span=siglen, adjust=False).mean()
    osc = erg - sig
    return osc


def calculate_smi(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        longlen: int = 20,
        shortlen: int = 5) -> pd.Series:
    """
    Calculate SMI Ergodic Indicator SMI (TSI) as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        longlen (int): Long length for TSI calculation (default: 20, min: 1)
        shortlen (int): Short length for TSI calculation (default: 5, min: 1)

    Returns:
        pd.Series: Series containing SMI (TSI) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if longlen < 1 or shortlen < 1:
        raise ValueError("Length parameters must be at least 1")

    def calculate_tsi(src, shortlen, longlen):
        """Calculate True Strength Index (TSI)."""
        delta = src.diff()
        abs_delta = delta.abs()
        ema_short = delta.ewm(span=shortlen, adjust=False).mean()
        ema_long = ema_short.ewm(span=longlen, adjust=False).mean()
        abs_ema_short = abs_delta.ewm(span=shortlen, adjust=False).mean()
        abs_ema_long = abs_ema_short.ewm(span=longlen, adjust=False).mean()
        tsi = 100 * ema_long / abs_ema_long.where(abs_ema_long != 0, np.nan)
        return tsi
    smi = calculate_tsi(data_close, shortlen, longlen)
    return smi


def calculate_smi_signal(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        longlen: int = 20,
        shortlen: int = 5,
        siglen: int = 5) -> pd.Series:
    """
    Calculate SMI Ergodic Indicator Signal line (EMA of TSI) as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        longlen (int): Long length for TSI calculation (default: 20, min: 1)
        shortlen (int): Short length for TSI calculation (default: 5, min: 1)
        siglen (int): Signal line EMA length (default: 5, min: 1)

    Returns:
        pd.Series: Series containing Signal (EMA of TSI) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if longlen < 1 or shortlen < 1 or siglen < 1:
        raise ValueError("Length parameters must be at least 1")

    def calculate_tsi(src, shortlen, longlen):
        """Calculate True Strength Index (TSI)."""
        delta = src.diff()
        abs_delta = delta.abs()
        ema_short = delta.ewm(span=shortlen, adjust=False).mean()
        ema_long = ema_short.ewm(span=longlen, adjust=False).mean()
        abs_ema_short = abs_delta.ewm(span=shortlen, adjust=False).mean()
        abs_ema_long = abs_ema_short.ewm(span=longlen, adjust=False).mean()
        tsi = 100 * ema_long / abs_ema_long.where(abs_ema_long != 0, np.nan)
        return tsi
    smi = calculate_tsi(data_close, shortlen, longlen)
    signal = smi.ewm(span=siglen, adjust=False).mean()
    return signal


def calculate_rvi(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 10,
        offset: int = 0) -> pd.Series:
    """
    Calculate Relative Volatility Index (RVI) as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for standard deviation calculation (default: 10, min: 1)
        offset (int): Offset for RVI values (default: 0, min: -500, max: 500)

    Returns:
        pd.Series: Series containing RVI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")

    if offset < -500 or offset > 500:
        raise ValueError("Offset must be between -500 and 500")
    stddev = data_close.rolling(window=length).std()
    change = data_close.diff()
    len_ema = 14
    upper = pd.Series(0.0, index=data_close.index)
    lower = pd.Series(0.0, index=data_close.index)
    upper[change <= 0] = stddev[change <= 0]
    lower[change > 0] = stddev[change > 0]
    upper_ema = upper.ewm(span=len_ema, adjust=False).mean()
    lower_ema = lower.ewm(span=len_ema, adjust=False).mean()
    rvi = upper_ema / (upper_ema + lower_ema).where(upper_ema +
                                                    lower_ema != 0, np.nan) * 100
    rvi = rvi.shift(offset)
    return rvi


def calculate_short_rci(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 10) -> pd.Series:
    """
    Calculate Short RCI line of the RCI Ribbon indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Length for Short RCI (default: 10, min: 1)

    Returns:
        pd.Series: Series containing Short RCI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")

    def calculate_rci(src, length):
        """Calculate Rank Correlation Index (RCI) for a given source and length."""
        rci = pd.Series(np.nan, index=src.index)
        for i in range(length - 1, len(src)):
            window = src.iloc[i - length + 1:i + 1]
            price_ranks = window.rank(ascending=True)
            time_ranks = pd.Series(range(1, length + 1), index=window.index)
            d_squared = (price_ranks - time_ranks) ** 2
            sum_d_squared = d_squared.sum()
            n = length
            max_d_squared = (n * (n**2 - 1)) / 6
            rci.iloc[i] = 100 * (1 - (sum_d_squared / max_d_squared)
                                 ) if max_d_squared != 0 else np.nan
        return rci
    return calculate_rci(data_close, length)


def calculate_middle_rci(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 30) -> pd.Series:
    """
    Calculate Middle RCI line of the RCI Ribbon indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Length for Middle RCI (default: 30, min: 1)

    Returns:
        pd.Series: Series containing Middle RCI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")

    def calculate_rci(src, length):
        """Calculate Rank Correlation Index (RCI) for a given source and length."""
        rci = pd.Series(np.nan, index=src.index)
        for i in range(length - 1, len(src)):
            window = src.iloc[i - length + 1:i + 1]
            price_ranks = window.rank(ascending=True)
            time_ranks = pd.Series(range(1, length + 1), index=window.index)
            d_squared = (price_ranks - time_ranks) ** 2
            sum_d_squared = d_squared.sum()
            n = length
            max_d_squared = (n * (n**2 - 1)) / 6
            rci.iloc[i] = 100 * (1 - (sum_d_squared / max_d_squared)
                                 ) if max_d_squared != 0 else np.nan
        return rci
    return calculate_rci(data_close, length)


def calculate_long_rci(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 50) -> pd.Series:
    """
    Calculate Long RCI line of the RCI Ribbon indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Length for Long RCI (default: 50, min: 1)

    Returns:
        pd.Series: Series containing Long RCI values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")

    def calculate_rci(src, length):
        """Calculate Rank Correlation Index (RCI) for a given source and length."""
        rci = pd.Series(np.nan, index=src.index)
        for i in range(length - 1, len(src)):
            window = src.iloc[i - length + 1:i + 1]
            price_ranks = window.rank(ascending=True)
            time_ranks = pd.Series(range(1, length + 1), index=window.index)
            d_squared = (price_ranks - time_ranks) ** 2
            sum_d_squared = d_squared.sum()
            n = length
            max_d_squared = (n * (n**2 - 1)) / 6
            rci.iloc[i] = 100 * (1 - (sum_d_squared / max_d_squared)
                                 ) if max_d_squared != 0 else np.nan
        return rci
    return calculate_rci(data_close, length)


def calculate_momentum(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 10) -> pd.Series:
    """
    Calculate Momentum indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for Momentum calculation (default: 10, min: 1)

    Returns:
        pd.Series: Series containing Momentum values (current source - source[len]), aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if len < 1:
        raise ValueError("Length must be at least 1")
    src = data_close
    mom = src - src.shift(len)
    return mom


def calculate_median(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 3) -> pd.Series:
    """
    Calculate Median line of the Median indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for median calculation (default: 3, min: 1)

    Returns:
        pd.Series: Series containing Median (50th percentile) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")
    source = (data_high + data_low) / 2
    median = source.rolling(window=length).quantile(0.5)

    return median


def calculate_upper_band(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 3,
        atr_length: int = 14,
        atr_mult: float = 2.0) -> pd.Series:
    """
    Calculate Upper Band of the Median indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for median calculation (default: 3, min: 1)
        atr_length (int): Lookback period for ATR calculation (default: 14, min: 1)
        atr_mult (float): Multiplier for ATR (default: 2.0, min: 0)

    Returns:
        pd.Series: Series containing Upper Band (Median + ATR * multiplier) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1 or atr_length < 1:
        raise ValueError("Length and ATR length must be at least 1")

    if atr_mult < 0:
        raise ValueError("ATR multiplier must be non-negative")
    source = (data_high + data_low) / 2
    median = source.rolling(window=length).quantile(0.5)
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    upper_band = median + (atr * atr_mult)

    return upper_band


def calculate_lower_band(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 3,
        atr_length: int = 14,
        atr_mult: float = 2.0) -> pd.Series:
    """
    Calculate Lower Band of the Median indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for median calculation (default: 3, min: 1)
        atr_length (int): Lookback period for ATR calculation (default: 14, min: 1)
        atr_mult (float): Multiplier for ATR (default: 2.0, min: 0)

    Returns:
        pd.Series: Series containing Lower Band (Median - ATR * multiplier) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1 or atr_length < 1:
        raise ValueError("Length and ATR length must be at least 1")

    if atr_mult < 0:
        raise ValueError("ATR multiplier must be non-negative")
    source = (data_high + data_low) / 2
    median = source.rolling(window=length).quantile(0.5)
    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    lower_band = median - (atr * atr_mult)

    return lower_band


def calculate_median_ema(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 3) -> pd.Series:
    """
    Calculate Median EMA of the Median indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for median and EMA calculation (default: 3, min: 1)

    Returns:
        pd.Series: Series containing EMA of Median values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")
    source = (data_high + data_low) / 2
    median = source.rolling(window=length).quantile(0.5)
    median_ema = median.ewm(span=length, adjust=False).mean()

    return median_ema


def calculate_mcginley_dynamic(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 14) -> pd.Series:
    """
    Calculate McGinley Dynamic indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for initial EMA and adjustment factor (default: 14, min: 1)

    Returns:
        pd.Series: Series containing McGinley Dynamic values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")
    if length < 1:
        raise ValueError("Length must be at least 1")
    src = data_close
    mg = pd.Series(np.nan, index=src.index)
    ema_initial = src.rolling(window=length).mean()
    mg.iloc[length - 1] = ema_initial.iloc[length - 1]
    for i in range(length, len(src)):
        if not np.isnan(mg.iloc[i - 1]) and mg.iloc[i - 1] != 0:
            ratio = src.iloc[i] / mg.iloc[i - 1]
            adjustment = length * (ratio ** 4)
            mg.iloc[i] = mg.iloc[i - 1] + (src.iloc[i] - mg.iloc[i - 1]) / \
                adjustment if adjustment != 0 else mg.iloc[i - 1]
    return mg


def calculate_lsma(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 25,
        offset: int = 0) -> pd.Series:
    """
    Calculate Least Squares Moving Average (LSMA) as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for linear regression (default: 25, min: 1)
        offset (int): Offset for LSMA values (default: 0)

    Returns:
        pd.Series: Series containing LSMA values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")

    def linreg(src, length):
        """Calculate linear regression value at the most recent point."""
        lsma = pd.Series(np.nan, index=src.index)
        x = np.arange(length)

        for i in range(length - 1, len(src)):
            y = src.iloc[i - length + 1:i + 1].values
            if len(y) == length and not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 1)
                lsma.iloc[i] = np.polyval(coeffs, length - 1)
        return lsma

    src = data_close
    lsma = linreg(src, length)

    lsma = lsma.shift(offset)

    return lsma


def calculate_fisher(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 9) -> pd.Series:
    """
    Calculate Fisher Transform line as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for highest/lowest calculation (default: 9, min: 1)

    Returns:
        pd.Series: Series containing Fisher Transform values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if len < 1:
        raise ValueError("Length must be at least 1")

    hl2 = (data_high + data_low) / 2

    value = pd.Series(0.0, index=hl2.index)
    fish1 = pd.Series(0.0, index=hl2.index)

    high_ = hl2.rolling(window=len).max()
    low_ = hl2.rolling(window=len).min()
    for i in range(1, len(hl2)):
        if high_.iloc[i] != low_.iloc[i]:
            raw_value = 0.66 * ((hl2.iloc[i] - low_.iloc[i]) / (
                high_.iloc[i] - low_.iloc[i]) - 0.5) + 0.67 * value.iloc[i - 1]
            value.iloc[i] = min(max(raw_value, -0.999), 0.999)
        else:
            value.iloc[i] = value.iloc[i - 1]

        if abs(value.iloc[i]) < 1:
            fish1.iloc[i] = 0.5 * np.log((1 + value.iloc[i]) /
                                         (1 - value.iloc[i])) + 0.5 * fish1.iloc[i - 1]
        else:
            fish1.iloc[i] = fish1.iloc[i - 1]

    return fish1


def calculate_trigger(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 9) -> pd.Series:
    """
    Calculate Trigger line (lagged Fisher Transform) as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for highest/lowest calculation (default: 9, min: 1)

    Returns:
        pd.Series: Series containing Trigger (lagged Fisher Transform) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if len < 1:
        raise ValueError("Length must be at least 1")

    hl2 = (data_high + data_low) / 2

    value = pd.Series(0.0, index=hl2.index)
    fish1 = pd.Series(0.0, index=hl2.index)

    high_ = hl2.rolling(window=len).max()
    low_ = hl2.rolling(window=len).min()
    for i in range(1, len(hl2)):
        if high_.iloc[i] != low_.iloc[i]:
            raw_value = 0.66 * ((hl2.iloc[i] - low_.iloc[i]) / (
                high_.iloc[i] - low_.iloc[i]) - 0.5) + 0.67 * value.iloc[i - 1]
            value.iloc[i] = min(max(raw_value, -0.999), 0.999)
        else:
            value.iloc[i] = value.iloc[i - 1]
        if abs(value.iloc[i]) < 1:
            fish1.iloc[i] = 0.5 * np.log((1 + value.iloc[i]) /
                                         (1 - value.iloc[i])) + 0.5 * fish1.iloc[i - 1]
        else:
            fish1.iloc[i] = fish1.iloc[i - 1]
    fish2 = fish1.shift(1)

    return fish2


def calculate_envelope_basis(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 20,
        exponential: bool = False) -> pd.Series:
    """
    Calculate Basis line of the Envelope indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for Basis calculation (default: 20, min: 1)
        exponential (bool): Use EMA instead of SMA for Basis (default: False)

    Returns:
        pd.Series: Series containing Basis (SMA or EMA) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if len < 1:
        raise ValueError("Length must be at least 1")

    src = data_close
    if exponential:
        basis = src.ewm(span=len, adjust=False).mean()
    else:
        basis = src.rolling(window=len).mean()

    return basis


def calculate_envelope_upper(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 20,
        percent: float = 10.0,
        exponential: bool = False) -> pd.Series:
    """
    Calculate Upper Band of the Envelope indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for Basis calculation (default: 20, min: 1)
        percent (float): Percentage deviation for Upper Band (default: 10.0)
        exponential (bool): Use EMA instead of SMA for Basis (default: False)

    Returns:
        pd.Series: Series containing Upper Band (Basis * (1 + percent/100)) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if len < 1:
        raise ValueError("Length must be at least 1")

    if percent < 0:
        raise ValueError("Percent must be non-negative")

    src = data_close
    if exponential:
        basis = src.ewm(span=len, adjust=False).mean()
    else:
        basis = src.rolling(window=len).mean()
    k = percent / 100.0
    upper = basis * (1 + k)

    return upper


def calculate_envelope_lower(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        len: int = 20,
        percent: float = 10.0,
        exponential: bool = False) -> pd.Series:
    """
    Calculate Lower Band of the Envelope indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        len (int): Lookback period for Basis calculation (default: 20, min: 1)
        percent (float): Percentage deviation for Lower Band (default: 10.0)
        exponential (bool): Use EMA instead of SMA for Basis (default: False)

    Returns:
        pd.Series: Series containing Lower Band (Basis * (1 - percent/100)) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if len < 1:
        raise ValueError("Length must be at least 1")

    if percent < 0:
        raise ValueError("Percent must be non-negative")
    src = data_close
    if exponential:
        basis = src.ewm(span=len, adjust=False).mean()
    else:
        basis = src.rolling(window=len).mean()
    k = percent / 100.0
    lower = basis * (1 - k)

    return lower


def calculate_donchian_upper(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 20,
        offset: int = 0) -> pd.Series:
    """
    Calculate Upper Band of the Donchian Channels indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for highest price calculation (default: 20, min: 1)
        offset (int): Offset for Upper Band values (default: 0)

    Returns:
        pd.Series: Series containing Upper Band (highest price) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")
    upper = data_high.rolling(window=length).max()
    upper = upper.shift(offset)

    return upper


def calculate_donchian_lower(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 20,
        offset: int = 0) -> pd.Series:
    """
    Calculate Lower Band of the Donchian Channels indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for lowest price calculation (default: 20, min: 1)
        offset (int): Offset for Lower Band values (default: 0)

    Returns:
        pd.Series: Series containing Lower Band (lowest price) values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")
    lower = data_low.rolling(window=length).min()
    lower = lower.shift(offset)

    return lower


def calculate_coppock_curve(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        wma_length: int = 10,
        long_roc_length: int = 14,
        short_roc_length: int = 11) -> pd.Series:
    """
    Calculate Coppock Curve indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        wma_length (int): Length for Weighted Moving Average (default: 10, min: 1)
        long_roc_length (int): Long period for Rate of Change (default: 14, min: 1)
        short_roc_length (int): Short period for Rate of Change (default: 11, min: 1)

    Returns:
        pd.Series: Series containing Coppock Curve values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if wma_length < 1 or long_roc_length < 1 or short_roc_length < 1:
        raise ValueError("Length parameters must be at least 1")
    src = data_close
    long_roc = src.pct_change(periods=long_roc_length) * 100
    short_roc = src.pct_change(periods=short_roc_length) * 100
    roc_sum = long_roc + short_roc
    weights = np.arange(1, wma_length + 1)
    return roc_sum.rolling(
        window=wma_length).apply(
        lambda x: np.sum(
            x *
            weights) /
        np.sum(weights) if not np.any(
            np.isnan(x)) else np.nan,
        raw=True)


def calculate_connors_rsi(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        lenrsi: int = 3,
        lenupdown: int = 2,
        lenroc: int = 100) -> pd.Series:
    """
    Calculate Connors RSI (CRSI) indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        lenrsi (int): Length for RSI calculation (default: 3, min: 1)
        lenupdown (int): Length for UpDown RSI calculation (default: 2, min: 1)
        lenroc (int): Length for Percent Rank of ROC (default: 100, min: 1)

    Returns:
        pd.Series: Series containing CRSI values (average of RSI, UpDown RSI, and Percent Rank), aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if lenrsi < 1 or lenupdown < 1 or lenroc < 1:
        raise ValueError("Length parameters must be at least 1")

    # Calculate standard RSI
    src = data_close
    delta = src.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=lenrsi).mean()
    avg_loss = loss.rolling(window=lenrsi).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ud = pd.Series(0.0, index=src.index)
    for i in range(1, len(src)):
        if src.iloc[i] == src.iloc[i - 1]:
            ud.iloc[i] = 0
        elif src.iloc[i] > src.iloc[i - 1]:
            ud.iloc[i] = 1 if ud.iloc[i - 1] <= 0 else ud.iloc[i - 1] + 1
        else:
            ud.iloc[i] = -1 if ud.iloc[i - 1] >= 0 else ud.iloc[i - 1] - 1
    ud_delta = ud.diff()
    ud_gain = ud_delta.where(ud_delta > 0, 0)
    ud_loss = -ud_delta.where(ud_delta < 0, 0)
    ud_avg_gain = ud_gain.rolling(window=lenupdown).mean()
    ud_avg_loss = ud_loss.rolling(window=lenupdown).mean()
    ud_rs = ud_avg_gain / ud_avg_loss.where(ud_avg_loss != 0, np.nan)
    updownrsi = 100 - (100 / (1 + ud_rs))
    roc = src.pct_change(periods=1) * 100
    percentrank = roc.rolling(window=lenroc).apply(lambda x: np.sum(x[-1] > x[:-1]) / (
        len(x) - 1) * 100 if not np.any(np.isnan(x)) and len(x) > 1 else np.nan, raw=True)
    crsi = (rsi + updownrsi + percentrank) / 3

    return crsi


def calculate_choppiness_index(
        data_open: pd.Series,
        data_high: pd.Series,
        data_low: pd.Series,
        data_close: pd.Series,
        length: int = 14,
        offset: int = 0) -> pd.Series:
    """
    Calculate Choppiness Index (CHOP) indicator as a pd.Series.

    Args:
        data_open (pd.Series): Input pandas Series with open price data
        data_high (pd.Series): Input pandas Series with high price data
        data_low (pd.Series): Input pandas Series with low price data
        data_close (pd.Series): Input pandas Series with close price data
        length (int): Lookback period for calculation (default: 14, min: 1)
        offset (int): Offset for CHOP values (default: 0, min: -500, max: 500)

    Returns:
        pd.Series: Series containing Choppiness Index values, aligned with input index
    """
    if not (
        data_open.index.equals(
            data_high.index) and data_high.index.equals(
            data_low.index) and data_low.index.equals(
                data_close.index)):
        raise ValueError("All input Series must have the same index")

    if length < 1:
        raise ValueError("Length must be at least 1")

    if offset < -500 or offset > 500:
        raise ValueError("Offset must be between -500 and 500")

    tr1 = data_high - data_low
    tr2 = np.abs(data_high - data_close.shift(1))
    tr3 = np.abs(data_low - data_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(window=length).sum()

    range_ = data_high.rolling(window=length).max(
    ) - data_low.rolling(window=length).min()

    ci = 100 * np.log10(atr_sum / range_.where(range_ !=
                        0, np.nan)) / np.log10(length)

    ci = ci.shift(offset)

    return ci
