import datetime as dt
import json
import os
import pandas as pd


def load_corpus(filename):
    module_dir = os.path.dirname(__file__)
    corpus = pd.read_csv(module_dir + "/" + filename)
    corpus.index = corpus.date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    corpus = corpus.dropna()
    corpus = corpus.drop(columns="date").body
    return corpus

def load_reddit_data(subreddits=None):
    if subreddits is None:
        config = _load_config()
        subreddits = config["subreddits"]

    submissions = pd.DataFrame()
    comments = pd.DataFrame()

    for subreddit in subreddits:
        module_dir = os.path.dirname(__file__)
        submissions_dir = module_dir+"/../data/{}_submissions".format(subreddit)
        comments_dir = module_dir+"/../data/{}_comments".format(subreddit)

        submissions = pd.concat([submissions, pd.read_csv(submissions_dir, index_col=0, encoding="utf8")])
        comments = pd.concat([comments, pd.read_csv(comments_dir, index_col=0, encoding="utf8")])

    submission = _prepare_reddit(submissions)
    comments = _prepare_reddit(comments)

    return submissions, comments


def load_btc_data():
    config = _load_config()
    interval = config["timeframe"]

    module_dir = os.path.dirname(__file__)
    btc = pd.read_csv(module_dir + "/../data/btc_ohlcv")
    btc.index = pd.DatetimeIndex(data=btc.date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")))
    btc = btc.drop(columns="date")
    btc = btc[~btc.index.duplicated()]
    btc = btc.sort_index()
    btc = btc.fillna(method="ffill")
    btc = btc.resample(interval).interpolate(method='linear')
    return btc


def _prepare_reddit(data):
    data = data.dropna()
    data.index = pd.DatetimeIndex(data=data.date.apply(dt.datetime.fromtimestamp))
    data = data.rename(columns={"date": "unix_date"})
    data = data.sort_index()
    return data

def _load_config():
    module_dir = os.path.dirname(__file__)
    with open(module_dir + "/../config.json", "r") as f:
        config = json.loads(f.read())
    return config

########################################################################################################################
