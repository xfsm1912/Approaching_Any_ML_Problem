from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

# create a corpus of sentences
corpus = [
    "hello, how are you?",
    "im getting bored at home. And you? What do you think?",
    "did you know about counts",
    "let's see if this works!",
    "YES!!!!"
]

# initialize CountVector
ctv = CountVectorizer(tokenizer=word_tokenize)

# fit the vectorizer on corpus
ctv.fit(corpus)

corpus_transformed = ctv.transform(corpus)

print(corpus_transformed)
print(ctv.vocabulary_)

