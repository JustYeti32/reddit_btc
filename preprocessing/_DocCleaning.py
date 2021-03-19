import re
from gensim.parsing.preprocessing import stem_text
from textblob import Word


def _clean_doc(doc: str, stopwords: set, duplicates: dict, lemmatize: bool, stem: bool, min_word_len: int):
    doc = _lower(doc)
    doc = _drop_del(doc)
    doc = _no_links(doc)
    doc = _no_breaks(doc)
    doc = _only_alphabetic(doc)
    doc = _no_spam(doc)
    doc = _dedup(doc, duplicates)

    if not stopwords:
        doc = _no_stopwords(doc, stopwords)
    if lemmatize:
        doc = _lemmatize(doc)
    if stem:
        doc = _stem(doc)

    doc = _no_short_words(doc, min_word_len)
    doc = _neat_spaces(doc)
    return doc

def _no_links(doc):
    doc = doc.split(" ")
    doc = [word for word in doc if "http" not in word]
    doc = [word for word in doc if ".com" not in word]
    doc = " ".join(doc)
    return doc

def _only_alphabetic(doc):
    doc = re.sub(r"[\d]", "", doc)
    doc = re.sub(r"[\W]|_", " ", doc)
    doc = re.sub(r"[^a-zA-Z]\s]", "", doc)
    return doc

def _no_stopwords(doc, _stopwords):
    doc = re.sub(_stopwords, "", doc)
    return doc

def _no_spam(doc):
    # remove words like yahahaha (3 or more repeating tuplets)
    doc = re.sub(r"(..)\1{2,}", r"\1\1", doc)

    # reduce repetitions of at least 3 letters to tuplet like mooon -> moon
    doc = re.sub(r"(.)\1{2,}", r"\1\1", doc)
    return doc

def _dedup(doc, _duplicates):
    for default in _duplicates:
        doc = re.sub(_duplicates[default], default, doc)
    return doc

def _no_breaks(doc):
    doc = re.sub("[\r\n\t\f]", " ", doc)
    return doc

def _lower(doc):
    doc = doc.lower()
    return doc

def _no_short_words(doc, min_word_len=2):
    doc = " ".join([word for word in doc.split(" ") if len(word) >= min_word_len])
    return doc

def _neat_spaces(doc):
    doc = re.sub(" {2,}", " ", doc)
    doc = doc.strip()
    return doc

def _stem(doc):
    doc = stem_text(doc)
    return doc

def _lemmatize(doc):
    doc = " ".join(Word(w).lemmatize() for w in doc.split(" "))
    return doc

def _drop_del(doc):
    if doc in ["[removed]", "[deleted]"]:
        doc = ""
    return doc

########################################################################################################################


