import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix as sk_conf
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.metrics import precision_recall_curve as sk_precrec

from statsmodels.tsa.stattools import grangercausalitytests

from matplotlib.cm import seismic


def scores(y_true, y_pred, y_prob):
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    global_scores = pd.DataFrame({"matthews": [mcc],
                                  "accuracy": [acc]
                                  })

    for class_, true_class_, pred_class_, prob_class_ in walk_classes(y_true, y_pred, y_prob):
        tp, fn, fp, tn = sk_conf(true_class_, pred_class_).flatten()
        class_prec = precision_score(true_class_, pred_class_)
        class_rec = recall_score(true_class_, pred_class_)
        class_auroc = roc_auc_score(true_class_, prob_class_)

        class_scores  = pd.DataFrame({f"class_{class_}_tp": [tp],
                                      f"class_{class_}_fp": [fp],
                                      f"class_{class_}_fn": [fn],
                                      f"class_{class_}_tn": [tn],
                                      f"class_{class_}_precision": [class_prec],
                                      f"class_{class_}_recall": [class_rec],
                                      f"class_{class_}_auroc": [class_auroc]
                                      })

        global_scores = pd.concat([global_scores, class_scores], axis=1)

    return global_scores

def pr_curve(y_true, y_pred, y_prob, plot=True):
    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title("PR-curve")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")

    for class_, true_class_, pred_class_, prob_class_ in walk_classes(y_true, y_pred, y_prob):
        class_prec_, class_rec_, _ = sk_precrec(true_class_, prob_class_)

        if plot:
            pos = sum(true_class_) / len(true_class_)
            line, = ax.plot([0,1],[pos, pos], linestyle="dotted")
            ap = round(average_precision_score(true_class_, prob_class_), 3)
            ax.plot(class_rec_, class_prec_, label=f"pos label: {class_}; AP: {ap}", color=line.get_color())

    if plot:
        ax.legend(loc="lower left")
    return

def roc_curve(y_true, y_pred, y_prob, plot=True):
    if plot:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.set_title("ROC-curve")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.plot([0,1],[0,1], label="no skill", color="k", linestyle="dotted")

    for class_, true_class_, pred_class_, prob_class_ in walk_classes(y_true, y_pred, y_prob):
        class_fpr_, class_tpr_, _ = sk_roc_curve(true_class_, prob_class_)

        if plot:
            auroc = round(roc_auc_score(true_class_, prob_class_), 3)
            ax.plot(class_fpr_, class_tpr_, label=f"pos label: {class_}; AUROC: {auroc}")

    if plot:
        ax.legend(loc="lower right")
    return

def walk_classes(y_true, y_pred, y_prob):
    classes = np.unique(y_true).astype(int)
    n_classes = len(classes)

    for class_ in classes:
        true_class_ = (y_true == class_).astype(int)
        pred_class_ = (y_pred == class_).astype(int)
        prob_class_ = y_prob[:, class_]
        yield class_, true_class_, pred_class_, prob_class_

def confusion(y_true, y_pred, plot=True):
    conf = sk_conf(y_true, y_pred, normalize="true")

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(conf, annot=True, cmap=seismic)
        ax.set_xlabel("prediction")
        ax.set_ylabel("true label")

    return conf

def causality(cause, effect, max_lag=25, sample_frequency="5min", plot=True):
    combined = pd.DataFrame(effect.rename("effect")).join(cause.rename("cause"), how="right")
    if sample_frequency is not None:
        combined = combined.resample(sample_frequency).last().dropna()

    cols = ["ssr_ftest", "ssr_chi2test", "lrtest"]
    ret = grangercausalitytests(combined, maxlag=max_lag, verbose=False)
    evaluation = [[ret[lag][0][col][1] for lag in range(1, max_lag + 1)] for col in cols]
    evaluation = pd.DataFrame(data=evaluation, index=cols).T.set_index(np.arange(1, max_lag + 1))

    if plot:
        fig, ax = plt.subplots(figsize=(18,5))
        evaluation.plot(ax=ax)

    return evaluation, combined

########################################################################################################################








