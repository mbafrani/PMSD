import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error


def get_fct_format(xdata, ydata):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
    ax.scatter(xdata, ydata, color='blue', linewidth=1)
    ax.set_xlabel(xdata.name)
    ax.set_ylabel(ydata.name)
    plt.show()
    prompt = "\nPlease try a good guess, choose one from following given options as the target function format by typing the number in front of corresponding function name and press 'Enter' \
       \n1. linear curve \
       \n2. Quadratic curve \
       \n3. Cubic curve \
       \n4. Quartic curve \
       \n5. Exponential curve \
       \n6. Sine curve \
       \n7. Cosine curve \
       \n8. Gaussian curve\
       \n9. Log curve\
       "
    label = [str(i) for i in range(1, 10)]
    fct_format = input(prompt)

    while fct_format not in label:
        print('You previous enter is nonsense, please select the right label!')
        fct_format = input(prompt)
    plt.close()
    return fct_format


def linear_curve(x, a, m):
    return a * x + m


def fit_linear_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(linear_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, linear_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*{}'.format(popt[0], xdata.name), m, rmse


def quadratic_curve(x, a, b, m):
    return a * x ** 2 + b * x + m


def fit_quadratic_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(quadratic_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, quadratic_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*{}**2+{}*{}'.format(popt[0],
                                   xdata.name, popt[1], xdata.name), m, rmse


def cubic_curve(x, a, b, c, m):
    return a * x ** 3 + b * x ** 2 + c * x + m


def fit_cubic_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(cubic_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, cubic_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*{}**3+{}*{}**2+{}*{}'.format(
        popt[0], xdata.name, popt[1], xdata.name, popt[2], xdata.name), m, rmse


def quartic_curve(x, a, b, c, d, m):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + m


def fit_quartic_curve(
        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(quartic_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, quartic_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*{}**4+{}*{}**3+{}*{}**2+{}*{}'.format(
        popt[0], xdata.name, popt[1], xdata.name, popt[2], xdata.name, popt[3], xdata.name), m, rmse


def exponential_curve(x, a, m):
    return a ** x + m


def fit_exponential_curve(
        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(
        exponential_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, exponential_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}**{}'.format(popt[0], xdata.name), m, rmse


def sine_curve(x, a, m):
    return a * np.sin(x) + m


def fit_sine_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(sine_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, sine_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*sin({})'.format(popt[0], xdata.name), m, rmse


def cosine_curve(x, a, m):
    return a * np.cos(x) + m


def fit_cosine_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(cosine_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, cosine_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*cos({})'.format(popt[0], xdata.name), m, rmse


def gaussian_curve(x, a, b, c, m):
    return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2))) + m


def fit_gaussian_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(gaussian_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(
        mean_squared_error(
            ytest, gaussian_curve(
                xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*exp(-({}-{}) **2 / 2*{}**2)'.format(
        popt[0], xdata.name, popt[1], popt[2]), m, rmse


def log_curve(x, a, m):
    return a * np.log(x) + m


def fit_log_curve(

        xdata,
        xtest,
        ydata,
        ytest,
        d_factor,
):
    popt, pcov = curve_fit(log_curve, xdata, ydata, maxfev=5000)
    rmse = np.sqrt(mean_squared_error(ytest, log_curve(xtest, *popt)))
    popt = [round(i / d_factor, 3) for i in popt]
    m = popt[-1]
    return '{}*log({})'.format(popt[0], xdata.name), m, rmse


def final_curve_fitting(xdata, xtest, ydata, ytest):
    """ get final curve fitting line """
    d_factor = len(xdata.columns)
    fct_dict = {
        '1': fit_linear_curve,
        '2': fit_quadratic_curve,
        '3': fit_cubic_curve,
        '4': fit_quartic_curve,
        '5': fit_exponential_curve,
        '6': fit_sine_curve,
        '7': fit_cosine_curve,
        '8': fit_gaussian_curve,
        '9': fit_log_curve}

    # step 1: get target function format
    fcts = {}
    for i in range(len(xdata.columns)):
        fct_format = get_fct_format(xdata[xdata.columns[i]], ydata)
        fcts[xdata.columns[i]] = fct_format

    # step2: fit curve
    fct_expression = ""
    m, rmse = 0, 0
    for fea, fct in fcts.items():
        func = fct_dict[fct]
        fct, constant_term, error = func(
            xdata[fea], xtest[fea], ydata, ytest, d_factor)
        if fct[0] == '-':
            fct_expression += fct
        else:
            fct_expression += '+{}'.format(fct)
        m += constant_term
        rmse += error
    equation = ydata.name + '=' + fct_expression + \
        '+' + str(round(m / len(xdata.columns), 3))

    return equation, rmse / d_factor
