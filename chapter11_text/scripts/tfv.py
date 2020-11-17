from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# create a corpus of sentences
corpus = [
    "hello, how are you?",
    "im getting bored at home. And you? What do you think?",
    "did you know about counts",
    "let's see if this works!",
    "YES!!!!"
]

# initialize TfidVectorizer with word_tokenize from nltk
# as the tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize)

# fit the vectorizer on corpus
tfv.fit(corpus)

corpus_transformed = tfv.transform(corpus)
print(corpus_transformed)
