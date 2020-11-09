import pandas as pd

from sklearn import tree
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# this is our global size of label text
# on the plots
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

df = pd.read_csv("./winequality-red.csv")

# a mapping dictionary that maps the quality values from 0 to 5
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}

df.loc[:, "quality"] = df.quality.map(quality_mapping)

#################
# use sample with frac=1 to shuffle the dataframe
# we reset the indices since they change after
# shuffing the dataframe
#################
df = df.sample(frac=1).reset_index(drop=True)

# top 1000 rows are selected
# for training
df_train = df.head(1000)

# bottom 599 values are selected
# fro testing/validation
df_test = df.tail(599)

# import from scikit-learn
# initialize decision tree classifier class
# with a max_depth of 3

clf = tree.DecisionTreeClassifier(max_depth=7)

# choose the columns you want to train on
# these are the features for the model
cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]


# train the model on the provided features
# and mapped quality from before
clf.fit(df_train[cols], df_train.quality)

# generate predictions on the training set
train_predictions = clf.predict(df_train[cols])

# generate predictions on the test set
test_predictions = clf.predict(df_test[cols])

# calculate the accuracy of predictions on
# training data set
train_accuracy = metrics.accuracy_score(
    df_train.quality, train_predictions
)

# calculate the accuracy of predictions on
# test data set
test_accuracy = metrics.accuracy_score(
    df_test.quality, test_predictions
)

print(train_accuracy)
print(test_accuracy)

# initialize lists to store accuracies
# for training and test data
# we start with 50% accuracy
train_accuracies = [0.5]
test_accuracies = [0.5]

for detph in range(1, 25):
    clf = tree.DecisionTreeClassifier(max_depth=detph)

    cols = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]

    # train the model on the provided features
    # and mapped quality from before
    clf.fit(df_train[cols], df_train.quality)

    # generate predictions on the training set
    train_predictions = clf.predict(df_train[cols])

    # generate predictions on the test set
    test_predictions = clf.predict(df_test[cols])

    # calculate the accuracy of predictions on
    # training data set
    train_accuracy = metrics.accuracy_score(
        df_train.quality, train_predictions
    )

    # calculate the accuracy of predictions on
    # test data set
    test_accuracy = metrics.accuracy_score(
        df_test.quality, test_predictions
    )

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()
