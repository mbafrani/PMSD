import matplotlib.pyplot as plt
import numpy as np
import os


def plot_rmse(rmse_list, target_feature, path_to_file=None):
    """ plot all rmse value from trained models"""

    x = list(rmse_list.keys())
    y = list(rmse_list.values())
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    width = 0.75
    ind = np.arange(len(y))  # the x locations for the groups
    for i, v in enumerate(y):
        ax.text(v, i, str(round(v, 3)), color='blue', fontweight='bold')
    ax.barh(ind, y, width, color="blue")
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(x, minor=False)
    plt.xlabel("RMSE Value")
    plt.ylabel("Model")
    plt.title(
        "RMSE of different models in terms of feature '{}'".format(target_feature))
    if path_to_file:
        plt.savefig(
            os.path.join(
                path_to_file,
                '{}.png'.format(
                    target_feature.replace(
                        ' ',
                        ''))))
    else:
        plt.show()
    plt.close()

