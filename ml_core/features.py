"""Feature engineering + data split shared by model_trainer, model_tester and
AQICaller, so training and inference can never drift apart."""
import numpy as np
import pandas as pd

try:
    from ml_core.config import FEATURE_COLS, TARGET_COL
except ImportError:  # run directly from inside ml_core/
    from config import FEATURE_COLS, TARGET_COL

NUMERIC_FEATURES = [c for c in FEATURE_COLS if c != 'region']


def add_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Build the 19 features for every region on a STRICT HOURLY GRID.

    Each region is reindexed to every hour between its first and last
    timestamp, so .shift(24) always means "24 hours ago" (not "24 rows ago")
    even when the source data has holes. Holes become NaN, and rolling windows
    require a full window, so any row that touches a hole gets NaN features.

    training=True : also adds the next-hour target and drops incomplete rows.
    training=False: keeps all rows (AQICaller checks the last row itself).
    Output is sorted by (region, timestamp) with a fresh RangeIndex.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('h')
    df['region'] = df['region'].astype(str)
    df = (df.drop_duplicates(['region', 'timestamp'], keep='last')
            .sort_values(['region', 'timestamp']))

    frames = []
    for region, g in df.groupby('region', sort=True):
        g = g.set_index('timestamp')
        grid = pd.date_range(g.index.min(), g.index.max(), freq='h', name='timestamp')
        g = g.reindex(grid)
        g['region'] = region

        aqi = g['aqi_value']
        g['aqi_lag_0h'] = aqi                      # current hour
        for k in (1, 2, 3, 24):
            g[f'aqi_lag_{k}h'] = aqi.shift(k)
        g['temp_lag_1h'] = g['temperature_aqi'].shift(1)
        g['wind_lag_1h'] = g['wind_speed_aqi'].shift(1)

        # Rolling stats on aqi_lag_1h, full window required (no partial windows)
        lag1 = g['aqi_lag_1h']
        for w in (3, 6, 24):
            g[f'aqi_roll_mean_{w}h'] = lag1.rolling(w, min_periods=w).mean()
        g['aqi_roll_std_24h'] = lag1.rolling(24, min_periods=24).std()

        g['hour'] = grid.hour
        g['dayofweek'] = grid.dayofweek
        g['month'] = grid.month

        if training:
            g[TARGET_COL] = aqi.shift(-1)          # next hour
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    if training:
        out = out.dropna(subset=NUMERIC_FEATURES + [TARGET_COL]).reset_index(drop=True)
    return out


def assign_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> pd.Series:
    """Per-region chronological split: each region's oldest rows -> 'train',
    then 'val' (early stopping), newest -> 'test' (final scoring only).
    `df` must be sorted by (region, timestamp) - add_features output is."""
    rank = df.groupby('region', observed=True).cumcount()
    size = df.groupby('region', observed=True)['region'].transform('size')
    frac = rank / size
    split = np.where(frac >= 1 - test_frac, 'test',
                     np.where(frac >= 1 - test_frac - val_frac, 'val', 'train'))
    return pd.Series(split, index=df.index)