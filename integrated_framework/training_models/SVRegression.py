from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import numpy as np


def call_svr(xtrain, xtest, ytrain, ytest):
    """
    :return: the best equation information founded by support vector regression with a linear kernel
    """

    # predefined kernels
    # kernels = ['linear','poly','rbf']
    # only with linear kernel, the coef_and intercept_ are assessable

    kernels = ['linear']
    # feature scaling
    sc = StandardScaler()
    scaled_xtrain = sc.fit_transform(xtrain)
    scaled_xtest = sc.fit_transform(xtest)
    scaled_ytrain = sc.fit_transform(ytrain.values.reshape(-1, 1))
    scaled_ytest = sc.fit_transform(ytest.values.reshape(-1, 1))

    res = {}
    for k in kernels:
        svr = SVR(kernel=k)
        svr.fit(scaled_xtrain, scaled_ytrain.ravel())
        predicted_res_test = svr.predict(scaled_xtest)
        predicted_res_train = svr.predict(scaled_xtrain)
        # inverse tranform to calculate RMSE
        rmse_test = round(np.sqrt(mean_squared_error(
            sc.inverse_transform(scaled_ytest),
            sc.inverse_transform(predicted_res_test))),3)

        rmse_training = round(np.sqrt(mean_squared_error(
            sc.inverse_transform(scaled_ytrain),
            sc.inverse_transform(predicted_res_train))),3)

        r2_train = round(r2_score(ytrain, predicted_res_train),3)
        r2_test = round(r2_score(ytest, predicted_res_test),3)

        res[k] = [svr, (rmse_test, rmse_training), (r2_test, r2_train)]
    selected_kernel = min(res.items(), key=lambda x: x[1][1])[0]

    # for debug
    #     IVs = list(xtrain.columns)
    #     var_coef = res[selected_kernel][0].coef_
    #     intercept = res[selected_kernel][0].intercept_
    #     rmse = res[selected_kernel][1]
    #     model_summary = [var_coef,intercept,IVs,selected_kernel,rmse]
    #     return model_summary
    # for debug

    # since coef_and intercept_ are only accessible when using a linear
    # kernel
    if selected_kernel == 'linear':
        IVs = list(xtrain.columns)
        var_coef = res['linear'][0].coef_
        intercept = res['linear'][0].intercept_
        (rmse_test, rmse_training) = res['linear'][1]
        (r2_test, r2_train) = res['linear'][2]
        model_summary = [var_coef[0], intercept[0], IVs,
                         (rmse_test, rmse_training), (r2_test, r2_train), rmse_training]

        print(
            'This is rmse value on testing set of model Support vector regression with linear kernel',
            rmse_test)
        print(
            'This is rmse value on training set of model Support vector regression with linear kernel',
            rmse_training)
        print('This is R^2 score on testing set of model Support vector regression with linear kernel', r2_test)
        print('This is R^2 score on training set of model Support vector regression with linear kernel', r2_train)
        return model_summary
    return [selected_kernel, res[selected_kernel][1]]


# def call_svr_2(xtrain, xtest, ytrain, ytest):
#     """
#     :return: the best equation information founded by support vector regression with a linear kernel
#     """
#
#     # predefined kernels
#     # kernels = ['linear','poly','rbf']
#     # only with linear kernel, the coef_and intercept_ are assessable
#
#     svr = SVR(kernel='linear')
#     svr.fit(xtrain, ytrain)
#     predicted_res_test = svr.predict(xtest)
#     predicted_res_train = svr.predict(xtrain)
#     # inverse tranform to calculate RMSE
#     rmse_test = mean_squared_error(
#         ytest, predicted_res_test)
#
#     rmse_training = mean_squared_error(
#         ytrain,
#         predicted_res_train)
#
#     r2_train = r2_score(ytrain, predicted_res_train)
#     r2_test = r2_score(ytest, predicted_res_test)
#
#     IVs = list(xtrain.columns)
#     var_coef = svr.coef_
#     intercept = svr.intercept_
# return [var_coef[0], intercept[0], IVs, (rmse_test, rmse_training),
# (r2_test, r2_train), rmse_test]
