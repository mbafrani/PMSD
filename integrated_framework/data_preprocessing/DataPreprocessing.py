import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

def remove_missing_values(df):
    """
    :param df: SD-Log in data frame
    :return: a dataframe without missing values (eliminate NaN)
    """

    if df.isnull().values.any():
        df = df.fillna(0)
    return df


def remove_outliers(df, proportion=0.04):
    """

    :param df: SD-Log in data frame
    :param proportion: The proportion of outliers in the data set, default as 0.04
    :return: cleaned SD-Log without outliers by using isolationForest
    """

    # detect outliers
    clf = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=proportion,
        max_features=df.shape[1],
        random_state=374639)
    clf.fit(df)
    outliers_predicted = clf.predict(df)

    # remove outliers
    cleaned_data = df[np.where(outliers_predicted == 1, True, False)]
    return cleaned_data


def data_spliting(xdata, ydata, related_predictors=None):
    """

    :param xdata: predictors values in SD-Log
    :param ydata: process variable values in SD-Log
    :param related_predictors: related variables which have a strong correlation again the process variable
    :return: data in training and testing set
    """
    if related_predictors is None:
        x = xdata
    else:
        x = xdata[related_predictors]
    x_train, x_test, y_train, y_test = train_test_split(
        x, ydata, random_state=374639)
    return x_train, x_test, y_train, y_test


def remove_trivial_parameters(sd_log):
    """
    given sd_log, remove trivial parameters, i.e., the columns only contain a same constant value

    :param sd_log: given sd_log
    :return: clean sd_log without trivial parameters
    """
    sd_log = sd_log.loc[:, (sd_log != sd_log.iloc[0]).any()]
    return sd_log

