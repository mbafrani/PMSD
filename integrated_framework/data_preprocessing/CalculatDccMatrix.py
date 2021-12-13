import dcor


def calculate_dcc_matrix(xdata, ydata):
    """

    :param xdata: predictors values
    :param ydata: process variable values
    :return: the distance correlation coefficient matirx
    """
    dcc = dict()
    for col in xdata:
        dcorr = dcor.distance_correlation(xdata[col], ydata)
        dcc[col] = round(dcorr, 3)  # rounded by 3 decimals
    return dcc


def select_predictors(dcc, threshold=0.05):
    """

    :param dcc: the distance correlation coefficient matrix
    :param threshold: the pre-defined threshold to filter predictor variables in SD-Log
    :return: selected predictors
    """
    related_vars = []
    for key, val in dcc.items():
        if val == 1:
            related_vars.append(key)
            return related_vars
    related_vars = [key for key,
                           val in dcc.items() if val >= threshold]
    return related_vars
