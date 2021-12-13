import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def stepwise_regression(xdata, ydata, num_feature):
    """ bi-directional elimination, while adding a new feature also checking the significance of already added feature"""
    sffs = SFS(LinearRegression(),
               k_features=num_feature,
               forward=True,
               floating=True,
               scoring='neg_mean_squared_error',
               cv=0)
    sffs.fit(xdata, ydata)
    return list(sffs.k_feature_names_)


def auto_selection_based_regression(
        xdata,
        xtest,
        ydata,
        ytest,
        num_feature='best'):
    """

    :param xdata: training set of predictors
    :param xtest: testing set of predictors
    :param ydata: training set of target
    :param ytest: testing set of target
    :param num_feature: select mode
    :return: the best equation info founded by stepwise regression
    """
    involved_IVs = stepwise_regression(xdata, ydata, num_feature)
    # get final model
    final_model = LinearRegression().fit(xdata[involved_IVs], ydata)
    var_coef = final_model.coef_
    intercept = final_model.intercept_

    # calculate rmse based on testing set
    rmse = round(np.sqrt(
        mean_squared_error(
            ytest, final_model.predict(
                xtest[involved_IVs]))),3)

    rmse_training = round(np.sqrt(
        mean_squared_error(
            ydata, final_model.predict(
                xdata[involved_IVs]))),3)
    r2_training = round(r2_score(ydata, final_model.predict(
        xdata[involved_IVs])),3)
    r2_testing = round(r2_score(ytest, final_model.predict(
        xtest[involved_IVs])),3)

    model_summary = [var_coef, intercept, involved_IVs, (rmse, rmse_training), (r2_testing, r2_training), rmse_training]

    print('this is rmse value on testing set of model stepwise regression', rmse)
    print('this is rmse value on training set of model stepwise regression', rmse_training)
    print('this is r^2 score on testing set of model stepwise regression', r2_testing)
    print('this is r^2 score on training set of model stepwise regression', r2_training)

    return model_summary
