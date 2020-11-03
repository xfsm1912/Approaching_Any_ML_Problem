import numpy as np
from scipy import stats


def rank_mean(probas):
    """  Create mean predictions using ranks
    :param probas: 2-d array of probability values
    :return: mean ranks
    """
    ranked = []
    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)
    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)


probas = np.array(
    [[1, 4, 5],
    [2, 1, 4]]
)
ranked_example = rank_mean(probas)


