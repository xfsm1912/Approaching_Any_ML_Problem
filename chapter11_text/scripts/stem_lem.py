from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# initialize stemmer
stemmer = SnowballStemmer("english")

words = ["fishing", "fishes", " fished"]

for word in words:
    print(f"word={word}")
    print(f"stemmed_word={stemmer.stem(word)}")
    print(f"lemma={lemmatizer.lemmatize(word)}")
    print("")



