import gensim.downloader
import os

from gensim.models import Word2Vec, Phrases


def word2vec(model="glove-twitter-25"):
    APPROVED = ["glove-twitter-25", "fasttext-wiki-news-subwords-300"]
    if model not in APPROVED:
        print("not approved choice, may break models which take a w2v as input")

    print(f"Loading {model}...")
    w2v = gensim.downloader.load(model)
    print(f"...Finished loading")
    return w2v

def train_word2vec(corpus, vector_size=25, window=3, bigrams=False, save_as=None):
    module_path = os.path.dirname(__file__)
    model_path = module_path + "/models"

    if type(corpus) == str:
        w2v = Word2Vec.load(model_path + "/" + corpus)
    else:
        corpus = corpus.apply(lambda x: x.split(" ")).to_list()

        if bigrams:
            bigram_transformer = Phrases(corpus)
            corpus = bigram_transformer[corpus]

        w2v = Word2Vec(sentences=corpus, size=vector_size, window=window, min_count=0)

        if save_as:
            try:
                os.mkdir(model_path)
            except:
                pass
            w2v.save(model_path + "/" + save_as)
    return w2v.wv

########################################################################################################################