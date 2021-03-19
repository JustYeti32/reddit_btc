import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

import preprocessing._DocCleaning
import preprocessing._Embeddings
import preprocessing._Metadata
import preprocessing._SentimentAnalysis
import preprocessing._TopicDiscovery
import preprocessing._Labeling
import preprocessing._Financial

from data.loaders import load_corpus

from gensim.corpora.dictionary import Dictionary

from matplotlib.cm import gist_earth_r

from tqdm import tqdm


class RedditText():
    def __init__(self, corpus):
        # Load attributes  __________________________________________
        if type(corpus) == pd.Series:
            self.corpus = corpus
        else:
            self.corpus = load_corpus(corpus)

        split = self.corpus.apply(lambda x: x.split(" "))
        self.dictionary = Dictionary(split)
        self.bow = pd.Series([self.dictionary.doc2bow(doc) for doc in split], index=self.corpus.index)

        # load private attributes ___________________________________
        self._stopwords = _Metadata._get_stopwords()
        self._duplicates = _Metadata._get_duplicates()

        # load methods ______________________________________________
        self.word2vec = _Embeddings.word2vec
        self.train_word2vec = _Embeddings.train_word2vec
        self.kmeans = _TopicDiscovery.kmeans
        self.lda = _TopicDiscovery.lda
        self.more_context = _TopicDiscovery.more_context
        self.same_context = _TopicDiscovery.same_context
        self.sentiment = _SentimentAnalysis.sentiment

        # load private methods ______________________________________
        self._clean_doc = _DocCleaning._clean_doc

        # ___________________________________________________________

    def clean_corpus(self, stopwords=False, lemmatize=True, stem=False, min_count=3, min_doc_len=3, save_as=False):
        _duplicates = self._duplicates
        _stopwords = self._stopwords

        corpus = [self._clean_doc(doc, _stopwords, _duplicates, lemmatize, stem, min_count) for doc in tqdm(self.corpus, desc="clean documents")]
        corpus = pd.Series(corpus, index=self.corpus.index).rename("body")
        corpus = corpus.loc[corpus.apply(len) >= min_doc_len]
        corpus = corpus.drop_duplicates()

        if save_as:
            module_dir = os.path.dirname(__file__)
            print(os.path.dirname(__file__))
            corpus.to_csv(module_dir+"/../data/"+save_as)
        return RedditText(corpus)

    def track(self, buzzwords: list, lemmatize=True, stem=False, min_count=5):
        buzzwords = [self.dictionary.token2id[buzzword] for buzzword in buzzwords]

        counts = pd.DataFrame()
        for buzzword in buzzwords:
            count = self.bow.apply(lambda bag: sum([word[1] if word[0]==buzzword else 0 for word in bag]))
            count = count.rename(self.dictionary[buzzword])
            counts = pd.concat([counts, count], axis=1)

        counts = counts.set_index(self.corpus.index)
        return counts

    def cfs(self):
        cfs = [[self.dictionary[key], self.dictionary.cfs[key]] for key in self.dictionary.cfs]
        cfs = pd.DataFrame(data=cfs, columns=["token", "wordcount"])
        cfs = cfs.sort_values(by="wordcount")
        return cfs

    def dfs(self):
        dfs = [[self.dictionary[key], self.dictionary.dfs[key]] for key in self.dictionary.dfs]
        dfs = pd.DataFrame(data=dfs, columns=["token", "wordcount"])
        dfs = dfs.sort_values(by="wordcount")
        return dfs


class OHLCV:
    def __init__(self, data):
        self.data = data
        self._get_label = _Labeling.get_label
        self._returns = _Financial._returns
        self._percent_returns = _Financial._percent_returns

    def add_labels(self, return_threshold, return_time, plot=False):
        setattr(self, "label_threshold", return_threshold)
        setattr(self, "label_time", return_time)
        label, aligned_label, stopped_percent_returns = self._get_label(self.data.close, return_threshold, return_time)

        self.data = self.data.drop(columns="label", errors="ignore").join(label)
        self.data = self.data.drop(columns="aligned_label", errors="ignore").join(aligned_label)
        self.data = self.data.drop(columns="stopped_percent_returns", errors="ignore").join(stopped_percent_returns)

        if plot:
            fig, ax = plt.subplots(figsize=(18,5))
            ax.plot(label, label="labels")
            ax.legend(loc="lower left")

        return

    def add_returns(self, return_time, plot=False):
        setattr(self, "return_time", return_time)
        returns = self._returns(self.data.close, return_time)
        percent_returns = self._percent_returns(self.data.close, return_time)

        self.data = self.data.drop(columns="returns", errors="ignore").join(returns)
        self.data = self.data.drop(columns="percent_returns", errors="ignore").join(percent_returns)

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=((18,5)))
            sns.histplot(self.data.returns, ax=ax[0])
            sns.histplot(self.data.percent_returns, ax=ax[1])
            sns.histplot(self.data.stopped_percent_returns, ax=ax[2])

        return

    # TODO perhaps deprciate
    def label_return_correlation(self, label_smoothing=None):
        sc_label = (self.data.label - 1) * self.label_threshold
        sc_label = sc_label.rename("prediction label")

        if label_smoothing is not None:
            sc_label = sc_label.rolling(label_smoothing).mean()

        sc_label_aligned = sc_label.shift(1, freq=self.label_time).reindex(self.data.index)
        sc_label_aligned = sc_label_aligned.rename("ground truth label")
        correlations = pd.concat([sc_label, sc_label_aligned, self.data.percent_returns], axis=1).corr()

        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].plot(self.data.percent_returns, linewidth=1, label="percent returns")
        sc_label.plot(ax=ax[0], label="label to predict")
        ax[0].legend(loc="upper left")

        ax[1].plot(self.data.percent_returns, linewidth=1, label="percent returns")
        sc_label_aligned.plot(ax=ax[1], label="ground truth label")
        ax[1].legend(loc="upper left")

        ax[2].plot(self.data.stopped_percent_returns, linewidth=1, label="stopped_percent returns")
        ax[2].legend(loc="upper left")

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.suptitle("pearson correlation")
        sns.heatmap(correlations, annot=True, ax=ax, cmap=gist_earth_r)
        return


    def chart(self, interval="5min"):
        close = self.data.close.resample(interval, closed="right", label="right").last()
        open_ = self.data.open.resample(interval, closed="right", label="right").first()
        high = self.data.high.resample(interval, closed="right", label="right").max()
        low = self.data.low.resample(interval, closed="right", label="right").min()
        volume = self.data.volume.resample(interval, closed="right", label="right").sum()

        chart = pd.concat([open_, high, low, close, volume], axis=1)
        if len(chart.dropna()) != len(chart):
            print("Error: generated NaN values; perhaps interval too short.")
            return None
        else:
            return chart


    def candleplot(self, interval="1D"):
        mpf.plot(self.chart(interval), type="candle", style="charles", volume=True, figsize=(18, 5))
        return



sns.set_style("white")

########################################################################################################################