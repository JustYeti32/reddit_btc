from data.loaders import load_reddit_data
from preprocessing import RedditText

submissions, comments = load_reddit_data(["CryptoCurrencyTrading"])
reddit = RedditText(comments.body)
w2v = reddit.train_word2vec("testw2v")
print("")