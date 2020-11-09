import chapter4_evaluation_metrics.utils as utils
import matplotlib.pyplot as plt

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0.02638412, 0.11114267, 0.31620708,
          0.0490937, 0.0191491, 0.17554844,
          0.15952202, 0.03819563, 0.11639273,
          0.079377, 0.08584789, 0.39095342,
          0.27259048, 0.03447096, 0.04644807,
          0.03543574, 0.18521942, 0.05934905,
          0.61977213, 0.33056815]


precisions = []
recalls = []

thresholds = [
    0.0490937, 0.05934905, 0.079377,  0.08584789, 0.11114267,
    0.11639273,  0.15952202, 0.17554844, 0.18521942,
    0.27259048, 0.31620708, 0.33056815,  0.39095342, 0.61977213
]

for i in thresholds:
    temp_prediction = [1 if x >= i else 0 for x in y_pred]
    p = utils.precision(y_true=y_true, y_pred=temp_prediction)
    r = utils.recall(y_true=y_true, y_pred=temp_prediction)
    precisions.append(p)
    recalls.append(r)

plt.figure(figsize=(7, 7))
plt.plot(recalls, precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

