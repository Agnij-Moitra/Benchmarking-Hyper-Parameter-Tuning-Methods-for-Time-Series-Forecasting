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
    intermediate = 2 * data.rolling(window=n // 2).apply(
        lambda x: np.dot(x, np.arange(1, n // 2 + 1)) /
        np.sum(np.arange(1, n // 2 + 1)),
        raw=True
    ) - data.rolling(window=n).apply(
        lambda x: np.dot(x, np.arange(1, n + 1)) / np.sum(np.arange(1, n + 1)),
        raw=True
    )

    sqrt_n = int(np.sqrt(n))
    weights = np.arange(1, sqrt_n + 1)
    return intermediate.rolling(window=sqrt_n).apply(
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


def calculate_stochastic_rsi(
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

    # Calculate %K
    stoch_rsi_k = ((rsi - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100
    return stoch_rsi_k.rolling(window=smooth_k).mean()

    # Calculate %D (optional, not returned in this implementation)
    # stoch_rsi_d = stoch_rsi_k_smooth.rolling(window=smooth_d).mean()


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
    Calculate Price Momentum Oscillator (PMO) for a pandas Series.

    Args:
        data (pd.Series): Input pandas Series with numerical data
        n (int): Period for EMA calculation (default: 20)

    Returns:
        pd.Series: Series containing PMO values
    """
    = data - data.shift(1)
    _diff = (- .shift(1)) * 100
    return 10 * _diff.ewm(span=n, adjust=False).mean()


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
