import matplotlib.pyplot as plt


def plot_simulated_res(stocks, sd_log, dummy_stock=None):
    """
    compare simulated results and raw observation in sd-log
    :param stocks: dataframe contains simulated result
    :param sd_log: raw sd-log
    :return:
    """
    if dummy_stock is not None:
        for var in sd_log.columns:
            plt.figure(figsize=(15, 10))
            plt.plot(
                stocks['TIME'],
                stocks[var],
                label='simulated result',
                color='orange')
            plt.plot(
                stocks['TIME'],
                dummy_stock[var],
                label='dummy result',
                color='green')
            plt.plot(
                stocks['TIME'],
                sd_log[var],
                label='Raw data',
                color='blue')
            plt.title('Result comparision on feature {}'.format(var))
            plt.legend()
            plt.show()
    else:
        for var in sd_log.columns:
            plt.figure(figsize=(15, 10))
            plt.plot(
                stocks['TIME'],
                stocks[var],
                label='simulated result',
                color='orange')
            plt.plot(
                stocks['TIME'],
                sd_log[var],
                label='Raw data',
                color='blue')
            plt.title('Result comparision on feature {}'.format(var))
            plt.legend()
            plt.show()
