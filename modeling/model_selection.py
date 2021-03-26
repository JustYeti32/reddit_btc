from preprocessing import OHLCV

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.cm import seismic, Reds


class TimeCV:
    def __init__(self, n_folds=4, embargo=dt.timedelta(days=7)):
        self.n_folds = n_folds
        self.embargo = embargo

    def cross_validate(self, X, y, model, verbose=True):
        y_tests = []
        y_preds = []
        y_probs = []

        folds = self.get_folds(X, y, verbose)
        for i, X_train, X_test, y_train, y_test in folds:
            if verbose:
                print(f"Training on fold {i+1}/{self.n_folds} | Have: {len(y_train)} samples")

            model_c = clone(model)
            model_c.fit(X_train, y_train)

            if verbose:
                print(f"Predicting on fold {i+1}/{self.n_folds} | Have: {len(y_test)} samples")
                print(100*"-")

            y_pred = pd.Series(model_c.predict(X_test), index=X_test.index).rename("prediction")
            y_preds.append(y_pred)
            y_tests.append(y_test)
            try:
                y_prob = pd.DataFrame(model_c.predict_proba(X_test), index=X_test.index)
                y_probs.append(y_prob)
            except:
                y_probs = None

        return y_tests, y_preds, y_probs

    def get_folds(self, X, y, verbose):
        dates = X.index
        time_span = dates[-1] - dates[0]
        fold_time = time_span / self.n_folds

        test_fold_start = dates[0]
        test_fold_end = test_fold_start + fold_time
        for i in range(self.n_folds):
            X_train = X.loc[(X.index < test_fold_start - self.embargo) | (X.index > test_fold_end + self.embargo)]
            y_train = y.loc[(X.index < test_fold_start - self.embargo) | (X.index > test_fold_end + self.embargo)]

            X_test = X[test_fold_start:test_fold_end] # half open [ , )
            y_test = y[test_fold_start:test_fold_end]

            if i < self.n_folds - 1:
                X_test = X_test[:-1] # half open, except for last
                y_test = y_test[:-1]

            if verbose:
                print(f"Fetched train fold before {test_fold_start - self.embargo} and after {test_fold_end + self.embargo}")
                print(f"Fetched test fold from {X_test.index[0]} to {X_test.index[-1]}")

            scaler = MinMaxScaler()
            X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(data=scaler.transform(X_test),  columns=X_test.columns, index=X_test.index)

            test_fold_start += fold_time
            test_fold_end += fold_time

            yield i, X_train, X_test, y_train, y_test


class TradingSimulation(OHLCV):
    def __init__(self, ohlcv, investment=1000):
        super().__init__(ohlcv)
        self.investment = investment

    def simulate(self, signals, sltp=0.05, timeout="1D", plot=True):
        signals = self.prediction_to_trade_signal(signals)

        self.add_labels(sltp, timeout)
        roi_percent = pd.DataFrame(self.data.stopped_percent_returns).join(signals, how="right").dropna()
        percent_gains = roi_percent.stopped_percent_returns * roi_percent.signals
        portfolio = self.investment + (percent_gains * self.investment).cumsum().rename("portfolio")

        no_skill = roi_percent.stopped_percent_returns
        no_skill_portfolio = self.investment + (no_skill * self.investment).cumsum().rename("no_skill_portfolio")

        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(18, 10))
            close = self.data.close[portfolio.index.min(): portfolio.index.max()]
            close.plot(ax=ax[0], linewidth=0.5, color="k")
            ax[0].set_xlabel("")
            ax[0].set_title("close price")

            portfolio.plot(ax=ax[1], linewidth=2, color="k", label="portfolio")
            no_skill_portfolio.plot(ax=ax[1], linewidth=2, color="grey", label="no skill", linestyle="dashed")
            ax[1].set_title(f"returns on {self.investment}$ initial investment (always all in)")
            ax[1].set_ylabel("dollars")
            ax[1].legend(loc="upper left")

            for num, now in enumerate(portfolio.index):
                prev = portfolio.index[num - 1]

                if ((signals[now] == 1) & (signals[prev] != 1)):
                    # open trade
                    ax[0].scatter(now, close[now], color="g", marker="^", s=100)
                    ax[1].scatter(now, portfolio[now], color="g", marker="^", s=100)
                if ((signals[now] != 1) & (signals[prev] == 1)):
                    ax[0].scatter(now, close[now], color="r", marker="v", s=100)
                    ax[1].scatter(now, portfolio[now], color="r", marker="v", s=100)

        portfolio = pd.concat([portfolio, no_skill_portfolio], axis=1)
        return portfolio

    @staticmethod
    def prediction_to_trade_signal(signals):
        signals = (signals==2).apply(int) # buy when up-prediction
        return signals

def linear_importance(X, y, plot=True):
    X_train = MinMaxScaler().fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y)
    importance = pd.DataFrame(data=model.coef_, columns=X.columns).T
    importance["class_mean"] = importance.apply(abs).mean(axis=1)
    importance = importance.sort_values(by="class_mean", ascending=False)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(importance, annot=True, ax=ax, cmap=seismic, center=0)

    return importance

def decision_importance(X, y, plot=True):
    X_train = MinMaxScaler().fit_transform(X)
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, y)
    importance = pd.DataFrame(data=model.feature_importances_, index=X.columns, columns=["importance"])
    importance = importance.sort_values(by="importance", ascending=False)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(importance, annot=True, ax=ax, cmap=Reds)

    return importance

########################################################################################################################
