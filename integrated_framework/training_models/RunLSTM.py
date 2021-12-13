import pandas as pd
from math import sqrt
from numpy import concatenate
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    take a multivariate time series and frame it as a supervised learning dataset
    Arguments:
        data: dataset, sequence of obervations as a list or 2D numpy array
        n_in: number of lag observations as input, range in [1, len(data)]
        n_out: number of observations as output, range in [0,len(data)-1]
        dropnan: boolean whether or not to drop rows with NaN values
    Return:
        pandas dataframe for supervised learning
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # input sequences
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecase sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    framed_df = pd.concat(cols, axis=1)
    framed_df.columns = names
    # drop rows with NaN values
    if dropnan:
        framed_df.dropna(inplace=True)
    return framed_df


def lstm_preprocessing(data, target_feature, n_in, n_out):
    """preprocessing for long short term memory network"""
    all_features = list(data.columns)
    n_features = len(all_features)
    all_features.remove(target_feature)
    data = data[[target_feature] + all_features]
    values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)
    # drop columns we don't want to predict
    droped_columns = ['var{i}(t)'.format(i=x) for x in range(
        len([target_feature]) + 1, n_features + 1)]
    if n_out >= 1:
        for n in range(1, n_out):
            droped_columns += ['var{i}(t+{p})'.format(i=x, p=n) for x in
                               range(len([target_feature]) + 1, n_features + 1)]
    reframed.drop(droped_columns, axis=1, inplace=True)
    return reframed, scaler


def call_lstm(data, target_feature, n_in=1, n_out=1):
    """
    run long short term memory network if there is no such a best equation for a process variable
    :return: the rmse values and predicted values
    """
    n_features = len(data.columns)
    reframed_data, scaler = lstm_preprocessing(
        data, target_feature, n_in, n_out)
    # print(reframed_data)
    values = reframed_data.values
    # print(values)
    X, y = values[:, :-
                  n_out *
                  len([target_feature])], values[:, -
                                                 n_out *
                                                 len([target_feature])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=374639)
    #     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train = X_train.reshape((X_train.shape[0], n_in, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_in, n_features))
    #     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=72,
        validation_data=(
            X_test,
            y_test),
        verbose=0,
        shuffle=False)
    # plot history
    #     pyplot.plot(history.history['loss'], label='train')
    #     pyplot.plot(history.history['val_loss'], label='test')
    #     pyplot.legend()
    #     pyplot.show()

    # make a prediction
    yhat = model.predict(X_test)
    # print(yhat.shape)
    X_test = X_test.reshape((X_test.shape[0], n_in * n_features))
    # print(X_test)

    # invert scaling for forecast
    inv_yhat = concatenate(
        (yhat, X_test[:, -X_test.shape[-1] + 1:]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    y_test = y_test.reshape((len(y_test), 1))
    inv_y = concatenate(
        (y_test, X_test[:, -X_test.shape[-1] + 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    return rmse, inv_yhat
