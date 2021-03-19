import pandas as pd


def _returns(close: pd.Series, return_time: str):
    """returns[now] = close[now] - close[now-return_time]"""
    past_close = close.shift(1, freq=return_time).reindex(close.index)
    returns = (close - past_close).rename("returns")
    return returns


def _percent_returns(close: pd.Series, return_time: str):
    """returns[now] = close[now] / close[now-return_time] - 1"""
    past_close = close.shift(1, freq=return_time).reindex(close.index)
    percent_returns = (close / past_close - 1).rename("percent_returns")
    return percent_returns


########################################################################################################################