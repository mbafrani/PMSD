import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV


def lasso_regression(xdata, xtest, ydata, ytest):
    """
    :return: the best equation information founded by lasso regression
    """
    # find the best alpha
    lassocv = LassoCV(cv=5)
    lassocv.fit(xdata, ydata)
    alpha = lassocv.alpha_

    # train a lasso model
    lasso_model = Lasso(alpha=alpha, fit_intercept=True)
    lasso_model.fit(xdata, ydata)
    rmse = round(np.sqrt(mean_squared_error(ytest, lasso_model.predict(X=xtest))),3)

    rmse_training = round(np.sqrt(
        mean_squared_error(
            ydata, lasso_model.predict(xdata))),3)

    var_coef = lasso_model.coef_
    intercept = lasso_model.intercept_
    IV = list(xdata.columns)

    r2_score_training = round(lasso_model.score(xdata, ydata),3)
    r2_score_test = round(lasso_model.score(xtest, ytest),3)

    print('this is rmse value on testing set of model lasso', rmse)
    print('this is rmse value on training set of model lasso', rmse_training)
    print('this is r2 value on testing set of model lasso', r2_score_test)
    print('this is r2 value on training set of model lasso', r2_score_training)

    # plot result
    #     y_test_predict = lasso_model.predict(xtest)
    #     plt.scatter(y_test_predict,ytest,marker='o')
    #     plt.xlabel('Predicted value')
    #     plt.ylabel('real value')
    #     plt.title('Prediction VS Real based on lasso Regression')
    #     plt.show()
    model_summary = [var_coef, intercept, IV, (rmse, rmse_training), (r2_score_test, r2_score_training), rmse_training]

    return model_summary


def ridge_regression(xdata, xtest, ydata, ytest):
    """

    :return: the best equation information founded by ridge regression
    """
    # find the best alpha
    ridgecv = RidgeCV(cv=5)
    ridgecv.fit(xdata, ydata)
    alpha = ridgecv.alpha_
    # train a ridge model
    ridge_model = Ridge(alpha=alpha, fit_intercept=True)
    ridge_model.fit(xdata, ydata)
    rmse = round(np.sqrt(mean_squared_error(ytest, ridge_model.predict(X=xtest))),3)

    rmse_training = round(np.sqrt(
        mean_squared_error(
            ydata, ridge_model.predict(xdata))),3)

    var_coef = ridge_model.coef_
    intercept = ridge_model.intercept_
    IVs = list(xdata.columns)

    r2_score_training = round(ridge_model.score(xdata, ydata),3)
    r2_score_test = round(ridge_model.score(xtest, ytest),3)

    print('this is rmse value on testing set of model ridge', rmse)
    print('this is rmse value on training set of model ridge', rmse_training)
    print('this is r2 value on testing set of model ridge', r2_score_test)
    print('this is r2 value on training set of model ridge', r2_score_training)

    # plot result
    #     y_test_predict = ridge_model.predict(xtest)
    #     plt.scatter(y_test_predict,ytest,marker='o')
    #     plt.xlabel('Predicted value')
    #     plt.ylabel('real value')
    #     plt.title('Prediction VS Real based on ridge Regression')
    #     plt.show()
    model_summary = [var_coef, intercept, IVs, (rmse, rmse_training), (r2_score_test, r2_score_training), rmse_training]

    return model_summary


def elastic_net(xdata, xtest, ydata, ytest):
    """
    :return:  the best equation information founded by elastic net
    """
    # find the best alpha
    elastic_cv = ElasticNetCV(cv=5)
    elastic_cv.fit(xdata, ydata)
    alpha = elastic_cv.alpha_
    l1_ratio = elastic_cv.l1_ratio_
    # train a elastic net
    elastic_net = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True)
    elastic_net.fit(xdata, ydata)
    rmse = round(np.sqrt(mean_squared_error(ytest, elastic_net.predict(X=xtest))),3)

    rmse_training = round(np.sqrt(
        mean_squared_error(
            ydata, elastic_net.predict(xdata))),3)

    var_coef = elastic_net.coef_
    intercept = elastic_net.intercept_
    IVs = list(xdata.columns)

    r2_score_training = round(elastic_net.score(xdata, ydata),3)
    r2_score_test = round(elastic_net.score(xtest, ytest),3)

    print('this is rmse value on testing set of model elastic net', rmse)
    print('this is rmse value on training set of model elastic net', rmse_training)
    print('this is r2 value on testing set of model elastic net', r2_score_test)
    print(
        'this is r2 value on training set of model elastic net',
        r2_score_training)

    # plot result
    #     y_test_predict = elastic_net.predict(xtest)
    #     plt.scatter(y_test_predict,ytest,marker='o')
    #     plt.xlabel('Predicted value')
    #     plt.ylabel('real value')
    #     plt.title('Prediction VS Real based on Elastic Net')
    #     plt.show()
    model_summary = [var_coef, intercept, IVs, (rmse, rmse_training), (r2_score_test, r2_score_training), rmse_training]

    return model_summary
