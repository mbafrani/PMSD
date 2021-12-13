import pandas as pd
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def fit_linear_regression(xdata, ydata):
    """

    :param xdata: predictors values
    :param ydata: target values
    :return: rss and r^2 score
    """
    linear_model_k = LinearRegression(fit_intercept=True)
    linear_model_k.fit(xdata, ydata)
    RSS = mean_squared_error(
        ydata, linear_model_k.predict(xdata)) * len(ydata)
    R_squared = linear_model_k.score(xdata, ydata)
    return RSS, R_squared


# best subset selection algorithm
def best_subset_selection( xdata, xtest, ydata, ytest):
    """

    :param xdata: training set of predictors
    :param xtest: testing set of predictors
    :param ydata: training set of process variable
    :param ytest: testing set of precess variable
    :return: equation information discovered by best subset selection
    """
    k = len(xdata.columns)
    RSS_list, R_squared_list, IV_list = [], [], []
    numb_features = []
    # looping over the entire IVs in xdata
    for k in range(1, k + 1):
        for combo in itertools.combinations(xdata.columns, k):
            tmp_result = fit_linear_regression(
                xdata[list(combo)], ydata)
            RSS_list.append(tmp_result[0])
            R_squared_list.append(tmp_result[1])
            IV_list.append(combo)
            numb_features.append(len(combo))

    # store in a dataframe
    df = pd.DataFrame({'numb_features': numb_features,
                       'RSS': RSS_list,
                       'R_squared': R_squared_list,
                       'features': IV_list})
    df_min = df[df.groupby('numb_features')[
        'RSS'].transform(min) == df['RSS']]  #minimal mean squared error
    # df_max = df[df.groupby('numb_features')[
    #     'R_squared'].transform(max) == df['R_squared']]

    # select the best model with AIC
    n = len(ydata)
    res_df = df_min.copy()
    res_df['AIC'] = 2 * res_df['numb_features'] + \
        n * np.log(res_df['RSS'] / n)
    x = res_df['AIC'].min()
    involved_IVs = res_df.loc[res_df['AIC'] == x]['features']
    # fetch involved IVs
    involved_IVs = [j for i in involved_IVs for j in i]
    # print('this is involved ivs', involved_IVs)

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

    print('this is the RMSE value on testing data set of model LR', rmse)
    print('this is the RMSE value on training data set of model LR',rmse_training)
    print('this is the r2_testing value of model LR', r2_testing)
    print('this is the r2_training value of model LR', r2_training)

    model_summary = [var_coef, intercept, involved_IVs, (rmse, rmse_training), (r2_testing, r2_training), rmse_training]

    # plot result
    #     y_test_predict = final_model.predict(xtest[involved_IVs])
    #     plt.scatter(y_test_predict,ytest,marker='o')
    #     plt.xlabel('Predicted value')
    #     plt.ylabel('real value')
    #     plt.title('Prediction VS Real based on simple linear Regression')
    #     plt.show()

    return model_summary
