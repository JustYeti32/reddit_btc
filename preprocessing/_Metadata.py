import os
import json

from gensim.parsing.preprocessing import STOPWORDS


def _get_stopwords():
    config = _load_config()
    drop_stopwords = config["drop_stopwords"]
    add_stopwords = config["add_stopwords"]
    stopwords = set(STOPWORDS)

    for word in add_stopwords:
        stopwords.add(word)
    for word in drop_stopwords:
        stopwords.remove(word)

    stopwords = "|".join(stopwords)
    return stopwords

def _get_duplicates():
    config = _load_config()
    dups = config["duplicates"]
    dups = {dup[0]: "|".join(dup[1]) for dup in dups}
    return dups

def _load_config():
    module_dir = os.path.dirname(__file__)
    with(open(module_dir + "/config.json", "r")) as f:
        config = json.loads(f.read())
    return config

########################################################################################################################