import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, lower_bound: int, upper_bound: int) -> np.array:
    """ Generate a list of indicates: 1 -> buy, 0 -> hold, -1 -> sell.
    Args:
        df (pd.DataFrame): OHLCV

    Returns:
        pd.Series: 1, 0 or -1 with length of df
    """
    sma_lower = df.Close.rolling(lower_bound).mean().bfill()
    sma_upper = df.Close.rolling(upper_bound).mean().bfill()
    
    sma_lower = pd.concat((sma_lower, sma_lower.iloc[-1:])).values
    sma_upper = pd.concat((sma_upper, sma_upper.iloc[-1:])).values
    
    buy_condition = (sma_lower[:-1] <= sma_upper[:-1]) & (sma_lower[1:] > sma_upper[1:])
    sell_condition = (sma_lower[:-1] >= sma_upper[:-1]) & (sma_lower[1:] < sma_upper[1:])
    indicator = np.where(
        buy_condition,
        1,
        np.where(
            sell_condition,
            -1,
            0
        )
    )
    return indicator
