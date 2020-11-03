import pandas as pd
from sklearn import preprocessing

# read training data
train = pd.read_csv('../input/train.csv')

# read test data
test = pd.read_csv('../input/test.csv')

# create a fake target column for test data
# since this column doesn't exist
test.loc[:, "target"] = -1

# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)

# make a list of features we are interested in
# id and target is something we should not encode
features = [x for x in train.columns if x not in ["id", "target"]]

# loop over the feature list
for feat in features:
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEncoder()

    # note the trick here
    # since it is categorical data, we fillna with a string
    # and we convert all the data to string type
    # so, no matter its int or float, it's converted to string
    # int/float but categorical!!!
    temp_col = data[feat].fillna('NONE').astype(str).values

    # we can use fit_transform here as we do not
    # have any extra test data whtat we need to transform on separately
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)

# split the training and test data again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)

