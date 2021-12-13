from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np


def PolynomialRegression(degree=2, **kwargs):
    """build a pipeline"""

    return make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression(
            **kwargs))


def polynomial_regression(xdata, xtest, ydata, ytest, degree):
    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    xdata_poly = poly_features.fit_transform(xdata)
    poly_model = LinearRegression()
    poly_model.fit(xdata_poly, ydata)

    # predicting
    predicted_y = poly_model.predict(xdata_poly)  # predicted value on training data

    y_test_predict = poly_model.predict(poly_features.fit_transform(xtest))

    # evaluating
    rmse_train = round(np.sqrt(mean_squared_error(ydata, predicted_y)),3)
    rmse_test = round(np.sqrt(mean_squared_error(ytest, y_test_predict)),3)

    #     plot predicted results
    #     plt.scatter(y_test_predict,ytest,marker='o')
    #     plt.xlabel('Predicted value')
    #     plt.ylabel('real value')
    #     plt.title('Prediction VS Real based on Elastic Net')
    #     plt.show()

    var_coef = poly_model.coef_
    intercept = poly_model.intercept_
    feature_names = poly_features.get_feature_names(xdata.columns)[1:]

    return var_coef, intercept, feature_names, rmse_test, rmse_train
    # return var_coef, intercept, feature_names, rmse_test


def train_polynomial_model(xtrain, xtest, ytrain, ytest):
    """first get the optimal degree, and then train a polynomial regreesion model"""
    # Define the GridSearchCV parameters

    param_grid = {'polynomialfeatures__degree': np.arange(1, 6),  # pick the best degree from [1,2,3,4,5]
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    #     grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5,iid=False)
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5)
    grid.fit(xtrain, ytrain)
    model = grid.best_estimator_
    degree = model.steps[0][-1].degree  # get the optimal degree
    var_coef, intercept, feature_names, rmse_test, rmse_train= polynomial_regression(
        xtrain, xtest, ytrain, ytest, degree)

    print('this is rmse value on testing set of model Polynomial Regression', rmse_test)
    print('this is rmse value on training set', rmse_train)
    # print('this is r2 value on testing set of model Polynomial Regression', r2_testing)
    # print('this is r2 value on training set of model Poly Reg', r2_training)

    model_summary = [var_coef[1:], intercept, feature_names, (rmse_test, rmse_train), (None, None), rmse_train]

    return model_summary
