import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('./input/train.csv')

#mapping = {
#     "Freezing": 0,
#     "Warm": 1,
#     "Cold": 2,
#     "Boiling Hot": 3,
#     "Hot": 4,
#     "Lava Hot": 5
# }
#
# df.loc[:, "ord_2"] = df.ord_2.map(mapping)
#
# print(df.ord_2.value_counts())

# fill NaN values in ord_2 column
df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")

# initialize LabelEncoder
lbl_enc = preprocessing.LabelEncoder()

# fit label encoder and transform values on ord_2 column
# P.S: do not use this directly. fit first, then transform
df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)

df["ord_4"].value_counts()
