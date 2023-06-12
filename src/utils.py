from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def mape(true, pred):
    try:
        mape = 100.0 * np.mean(np.abs((pred - true) / (true)))
    except ZeroDivisionError:
        mape = 100.0 * np.mean(np.abs((pred - true) / (true + 0.1)))

    return mape