import numpy as np


def find_sentiment(sentence, pos, neg):
    """  This function returns sentiment of sentence
    :param sentence: sentence, a string
    :param pos: set of positive words
    :param neg: set of negative words
    :return: returns positive, negative or neutral sentiment
    """
    # split sentence by a space
    # "this is a sentence!" becomes:
    # #["this", "is" "a", "sentence!"]
    # #note that im splitting on all whitespaces
    # #if you want to split by space use .split("")
    sentence = sentence.split()

    # make sentence into a set
    sentence = set(sentence)

    # check number of common words with positive
    num_common_pos = len(sentence.intersection(pos))

    # check number of common words with negative
    num_common_neg = len(sentence.intersection(neg))

    # make conditions and return
    # see how return used eliminates if else
    if num_common_pos > num_common_neg:
        return "positive"
    if num_common_pos < num_common_neg:
        return "negative"
    return "neural"


def load_embeddings(word_index, embedding_file, vector_length=300):
    """
    A general function to create embedding matrix
    :param word_index: word:index dictionary
    :param embedding_file: path to embeddings file
    :param vector_length: length of vector
    :return:
    """

    max_features = len(word_index) + 1
    words_to_find = list(word_index.keys())
    more_words_to_find = []

    for wtf in words_to_find:
        more_words_to_find.append(wtf)
        more_words_to_find.append(str(wtf).capitalize())

    more_words_to_find = set(more_words_to_find)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embedding_index = dict(
        get_coefs(*o.strip().split(" "))
        for o in open(embedding_file)
        if o.split(" ")[0]
        in more_words_to_find
        and len(o) > 100
    )

    embedding_matrix = np.zeros((max_features, vector_length))
    for word, i in word_index.items():
        if i >= max_features:
            continue

        embedding_vector = embedding_index.get(word)
        if embedding_vector is None:
            embedding_vector = embedding_index.get(
                str(word).capitalize()
            )

        if embedding_vector is None:
            embedding_vector = embedding_index.get(
                str(word).upper()
            )

        if embedding_vector is not None and len(embedding_vector) == vector_length:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



