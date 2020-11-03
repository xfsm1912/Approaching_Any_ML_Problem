import numpy as np
from sklearn import impute

# create a random numpy array with 10 sample
# and 6 features and values ranging from 1 to 15
X = np.random.randint(1, 15, (10, 6))

# convert the array to float
X = X.astype(float)

# randomly assign 10 elements to NaN (missing)
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan

# use 2 nearest neighbours to fill na values
knn_imputer = impute.KNNImputer(n_neighbors=2)
X_imputer = knn_imputer.fit_transform(X)
print(X)
print(X_imputer)
