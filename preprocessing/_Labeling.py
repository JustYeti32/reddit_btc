import pandas as pd

from preprocessing._Financial import _percent_returns


def get_label(close: pd.Series, return_threshold: float, return_time: str):
    aligned_labels = close.rolling(return_time, closed="both").apply(lambda x: _triple_barrier_label(x, return_threshold)).rename("aligned_label")

    labels = aligned_labels.shift(-1, freq=return_time).reindex(close.index).rename("label")  # align with present

    stopped_percent_returns = _percent_returns(close, return_time)  # i.e. return of order placed at [now-return_time]
    stopped_percent_returns.loc[aligned_labels == 0] = - return_threshold
    stopped_percent_returns.loc[aligned_labels == 2] = + return_threshold
    stopped_percent_returns = stopped_percent_returns.rename("stopped_percent_returns")
    return labels, aligned_labels, stopped_percent_returns


def _triple_barrier_label(series, threshold_return):
    start = series[0]
    for price in series:
        if price / start - 1 >= threshold_return:
            return 2
        if price / start - 1 <= -threshold_return:
            return 0
    return 1

########################################################################################################################