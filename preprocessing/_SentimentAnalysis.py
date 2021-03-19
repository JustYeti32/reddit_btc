import pandas as pd
from textblob import TextBlob


def sentiment(corpus):
    sentiments = corpus.apply(lambda doc: TextBlob(doc).sentiment)
    polarity = sentiments.apply(lambda x: x.polarity).rename("polarity")
    subjectivity = sentiments.apply(lambda x: x.subjectivity).rename("subjectivity")
    analysis = pd.concat([polarity, subjectivity], axis=1)
    return analysis

########################################################################################################################