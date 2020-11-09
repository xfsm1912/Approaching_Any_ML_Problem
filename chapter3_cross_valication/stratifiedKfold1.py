import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('train.csv')

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = f

    df.to_csv("train_folds.csv", index=False)