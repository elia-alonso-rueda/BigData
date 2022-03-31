# pip install -U scikit-learn
import sklearn

# LOADING THE DATA SET
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

print(list(newsgroups_train.target_names))

# EXTRACTING TOPICS
