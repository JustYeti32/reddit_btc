import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis.gensim
import os
import numpy as np

from matplotlib.cm import cividis, gist_earth_r
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore


def kmeans(dictionary, w2v, n_clusters=5, buzzwords=[], plot=True, verbose=True, topn=10):
    if type(buzzwords) != list:
        buzzwords = [buzzwords]

    words = list(dictionary.token2id.keys())

    known_words = [word for word in words if word in w2v.vocab]
    unknown_words = [word for word in words if word not in w2v.vocab]

    known_buzzwords = [word for word in buzzwords if word in w2v.vocab]
    unknown_buzzwords = [word for word in buzzwords if word not in w2v.vocab]

    if verbose:
        print(f"{len(known_words)} words in dictionary could be matched with w2v")
        print(f"{len(unknown_words)} words in dictionary could not be matched with w2v")
        print(f"compute a clustering only with respect to matched words")
        print("")
        print(f"{len(known_buzzwords)} buzzwords could be matched with w2v")
        print(f"{len(unknown_buzzwords)} buzzwords could not be matched with w2v")

    if verbose:
        print(f"compute clustering with {n_clusters} clusters")
    X = np.vstack([w2v[word] for word in known_words])
    cluster_model = KMeans(n_clusters)
    cluster_model.fit(X)

    if plot:
        if verbose:
            print("compute pca for plotting")

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(X)
        centers_pca = pca.transform(cluster_model.cluster_centers_)

        fig, ax = plt.subplots(figsize=(24, 16))
        ax.set_title("{} clusters in word embeddings".format(n_clusters))
        ax.scatter(x_pca[:, 0], x_pca[:, 1], c=cluster_model.labels_, cmap=cividis, s=15, alpha=0.75)
        for i in range(n_clusters):
            ax.scatter(centers_pca[i, 0], centers_pca[i, 1], s=1000, facecolor="w", edgecolor="k", alpha=0.5, linewidth=10)

            if verbose:
                print(f"Most similar words to centroid {i}")
                for sim_word, similarity in w2v.most_similar([cluster_model.cluster_centers_[i, :]], topn=topn):
                    print(4 * " ", "|", 3 * " ", f"cosine similarity: {round(similarity, 4)} | word: \"{sim_word}\"")

        if buzzwords:
            for buzzword in known_buzzwords:
                belonging = cluster_model.predict(w2v[buzzword].reshape(1,-1))
                buzzwords_pca = pca.transform(w2v[buzzword].reshape(1,-1))

                text = buzzword + f" | {belonging[0]}"
                ax.annotate(text, (buzzwords_pca[0,0], buzzwords_pca[0,1]), backgroundcolor="bisque")
    else:
        pca = None

    return cluster_model, pca

def lda(corpus, num_topics=5, save_as=None, load=None, verbose=True):
    module_path = os.path.dirname(__file__)
    model_path = module_path + "/models"

    if verbose:
        print("prepare data")
    corpus = corpus.apply(lambda x: x.split(" "))
    dictionary = Dictionary(corpus)
    bow = [dictionary.doc2bow(doc) for doc in corpus]

    if type(load) == str:
        if verbose:
            print("loading lda")
        lda = LdaMulticore.load(model_path + "/" + load)
    else:
        if verbose:
            print("training lda")
        lda = LdaMulticore(bow, num_topics=num_topics)
        if save_as:
            try:
                os.mkdir(model_path)
            except:
                pass

            lda.save(model_path + "/" + save_as)
    if verbose:
        print("generate visualization")
    vis = pyLDAvis.gensim.prepare(lda, bow, dictionary)
    return lda, vis

def same_context(w2v, buzzwords, plot=True, annot=False):
    similarities = [[w2v.similarity(word_a, word_b) for word_a in buzzwords] for word_b in buzzwords]
    similarities = pd.DataFrame(similarities, index=buzzwords, columns=buzzwords)

    if plot:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_title("similarities according to {}".format("w2v"))
        sns.heatmap(similarities, annot=annot, ax=ax, cmap=gist_earth_r)
    return similarities

def more_context(w2v, buzzwords, topn=5, verbose=True):
    if verbose:
        print("context modeled by embedding")
        print(40 * "-")

    sim_words_dict = {}
    for word in buzzwords:
        n_most_similar = w2v.most_similar(word, topn=topn)
        sim_words_dict.update({word: n_most_similar})
        if verbose:
            print(4 * " ", "|", f"other words in the context of \"{word}\":")
            for sim_word, similarity in n_most_similar:
                print(4 * " ", "|", 3 * " ", f"similarity: {round(similarity, 4)} | word: \"{sim_word}\"")
                print(4 * " ", "|", 3 * " ", 26 * "-")
    return sim_words_dict

########################################################################################################################

try:
    pyLDAvis.enable_notebook()
except:
    pass

########################################################################################################################