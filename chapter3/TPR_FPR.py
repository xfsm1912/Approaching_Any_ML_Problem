import chapter3.utils as utils
import matplotlib.pyplot as plt
from sklearn import metrics

# empty lists to store tpr
# and fpr values

tpr_list = []
fpr_list = []

# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,  0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [
    0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
    0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
    0.85, 0.15, 0.99
]

# handmade thresholds
thresholds = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0
]

# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given thresholds
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tpr
    temp_tpr = utils.tpr(y_true, temp_pred)
    # calculate fpr
    temp_fpr = utils.fpr(y_true, temp_pred)
    # append tpr and fpr to lists
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)


print(metrics.roc_auc_score(y_true, y_pred))

plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()



