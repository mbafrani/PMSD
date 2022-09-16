import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.signal import argrelmin, argrelmax
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax


def cp_detection_binary_segmentation(points,  show_plot=False, save_plot=False, outputpath=None, pen=3):
    # Changepoint detection with the Binary Segmentation search method
    model = "rbf"
    n = len(points)
    algo = rpt.Binseg(model=model).fit(points)
    # my_bkps = algo.predict(n_bkps=2) # if number of change poins is known a priori
    my_bkps = algo.predict(pen=pen)
    # show results
    if show_plot or save_plot:
        fig, (ax,) = rpt.display(points, my_bkps, figsize=(10, 6))
        if show_plot:
            plt.show()
        elif save_plot:
            plt.tight_layout()
            plt.savefig(outputpath, dpi=300)
    return my_bkps


def cp_detection_KSWIN(points, window_size=100, stat_size=40, period=None, show_plot=False, save_plot=False, outputpath=None):
    # Kolmogorov-Smirnov test
    from skmultiflow.drift_detection import KSWIN
    if period:
        window_size, stat_size = get_windows_size(period)
    kswin = KSWIN(alpha=0.05, window_size=window_size, stat_size=stat_size)
    # Store detection
    detections = []
    p_values = {}
    # Process stream via KSWIN and print detections
    for i in range(len(points)):
        batch = points[i]
        kswin.add_element(batch)
        if kswin.detected_change():
            print("\rIteration {}".format(i))
            print("\r KSWINReject Null Hyptheses")
            detections.append(i)
            p_values[i] = kswin.p_value
    print("Number of detections: " + str(len(detections)))
    if show_plot or save_plot:
        rpt.show.display(points, detections, figsize=(10, 6))
        plt.title('Change Point Detection: Kolmogorov-Smirnov Windowing')
        if show_plot:
            plt.show()
        elif save_plot:
            plt.tight_layout()
            plt.savefig(outputpath)
    return detections


def cp_detection_ADWIN(points):
    from skmultiflow.drift_detection.adwin import ADWIN
    adwin = ADWIN()
    detections = []
    # Adding stream elements to ADWIN and verifying if drift occurred
    for i in range(len(points)):
        adwin.add_element(points[i])
        if adwin.detected_change():
            detections.append(i)
            print('Change detected in data: ' + str(points[i]) + ' - at index: ' + str(i))
    rpt.show.display(points, detections, figsize=(10, 6))
    plt.title('Change Point Detection: ADWIN')
    plt.show()
    return detections


def cp_detection_PELT(points, show_plot=False, save_plot=False, clustering=False, outputpath=None, pen=1):
    # change point detection using pelt search method
    model = 'rbf'  # "l1" "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(points)
    my_bkps = algo.predict(pen=pen)

    # show results
    if show_plot or save_plot or clustering:
        fig, (ax,) = rpt.display(points, my_bkps, figsize=(10, 6))
        if show_plot:
            plt.show()
        elif save_plot:
            plt.tight_layout()
            plt.savefig(outputpath, dpi=300)

    return my_bkps


def decompostion_STL(series, period=None, title=''):
    stl = STL(series, period=period, robust=True)
    res_robust = stl.fit()
    fig = res_robust.plot()
    fig.text(0.1, 0.95, title, size=15, color='purple')
    plt.show()
    return stl


def tp_detection(series, period=None):  # series numpyarray
    """
    Get turning points by inspeccting trend cycle of series for finding local extremas
    :param series: numpy array
    :param period: period for detecting trend cycle in decomposition. If none take whole series as input
    :return: returns pandas series of turning points
    """
    if period is not None:
        stl = STL(series, period=period, robust=True)
        res_robust = stl.fit()
        trend_x = res_robust.trend
        series_x = pd.Series(trend_x)
    else:
        series_x = series

    N = 1  # number of iterations
    s_h = series_x.dropna().copy()  # make a series of Highs
    s_l = series_x.dropna().copy()  # make a series of Lows
    for i in range(N):
        s_h = s_h.iloc[argrelmax(s_h.values)[0]]  # locate maxima
        s_l = s_l.iloc[argrelmin(s_l.values)[0]]  # locate minima
        s_h = s_h[~s_h.index.isin(s_l.index)]  # drop index that appear in both
        s_l = s_l[~s_l.index.isin(s_h.index)]  # drop index that appear in both
    res = pd.concat([s_h, s_l]).sort_index()

    return res


def get_windows_size(tw):
    # one of ['1H', '8H', '1D', '7D']
    # set windows size two two weeks depending on log, stat size one week
    if tw == '1H':
        window_size = 2 * 168
        stat_size = 168
    elif tw == '8H':
        window_size = 2 * 21
        stat_size = 21
    elif tw == '1D':
        window_size = 2 * 7
        stat_size = 7
    elif tw == '7D':
        window_size = 4
        stat_size = 1
    else:
        window_size = 100
        stat_size = 40

    return window_size, stat_size


def subseqeuence_clustering(sequence, changepoints, y_label='y', show_plot=False, save_plot=False, norm=False,
                            outputpath=None, title=None):
    """
    Clusters subsequences of time series indicated by the changepoints variable.
    Uses silhouette score to determine the number of clusters
    :param y_label: Name of y-label in plot
    :param norm: normlise data using MinMaxScaler
    :param sequence: np array of the time series
    :param changepoints: detected changepoints on which subseuqences are build
    :return:
    """

    sub_ids = []
    x_index = []
    X = []
    i = 0
    end_p = [len(sequence) - 1]
    if end_p[0] < changepoints[-1]:  # Pelt gives last point as changepoint for visualisation purposes. Handle that
        changepoints.pop()
    for cp in changepoints + end_p:# + end_p:
        X.append(sequence[i:cp])
        index = 'sub_' + str(i) + '_' + str(cp)
        sub_ids.append(index)
        x_index.append([x_id for x_id in range(i, cp + 1)])
        i = cp

    # Normalize the data (y = (x - min) / (max - min))
    if norm:
        X = TimeSeriesScalerMinMax().fit_transform(X)
    X = to_time_series_dataset(X)
    #  Find optimal # clusters by
    #  looping through different configurations for # of clusters and store the respective values for silhouette:
    sil_scores = {}
    for n in range(2, len(changepoints)+1):
        model_tst = TimeSeriesKMeans(n_clusters=n, metric="dtw", n_init=10)
        model_tst.fit(X)
        sil_scores[n] = (silhouette_score(X, model_tst.predict(X), metric="dtw"))
    if len(changepoints) == 2:
        opt_k = 2
    elif len(changepoints) == 1:
        opt_k = 2
    elif len(changepoints) == 0:
        opt_k = 1
    else:
        opt_k = max(sil_scores, key=sil_scores.get)
    print('Number of Clusters in subsequence clustering: ' + str(opt_k)+'; Silhouette Scores: '+str(sil_scores))

    model = TimeSeriesKMeans(n_clusters=opt_k, metric="dtw", n_init=10)
    labels = model.fit_predict(X)
    print(labels)


    # build helper df to map metrics to their cluster labels
    df_cluster = pd.DataFrame(list(zip(sub_ids, x_index, model.labels_)), columns=['metric', 'x_index', 'cluster'])
    cluster_metrics_dict = df_cluster.groupby(['cluster'])['metric'].apply(lambda x: [x for x in x]).to_dict()

    print('Plotting Clusters')
    #  plot changepoints as vertical lines
    for cp in changepoints:
        plt.axvline(x=cp, ls=':', lw=2, c='0.65')
    #  preprocessing for plotting cluster based
    x_scat = []
    y_scat = []
    cluster = []
    for index, row in df_cluster.iterrows():
        x_seq = row['x_index']
        x_scat.extend(x_seq)
        y_seq = sequence[x_seq[0]:x_seq[-1] + 1]
        y_scat.extend(y_seq)
        label_seq = [row['cluster']]
        cluster.extend(label_seq * len(x_seq))
        # plt.scatter(x_seq, y_seq, label=label_seq)
    # plotting cluster based
    x_scat = np.array(x_scat)
    y_scat = np.array(y_scat)
    for c in np.unique(cluster):
        i = np.where(cluster == c)
        plt.scatter(x_scat[i], y_scat[i], label="Cluster "+str(c))
    plt.legend()
    if title:
        plt.title(title)
    else:
        plt.title('Subsequence K-Means Clustering')
    plt.xlabel('Time index')
    plt.ylabel(y_label, fontsize=16)
    if show_plot:
        plt.show()
    if save_plot:
        plt.tight_layout()
        plt.savefig(outputpath, dpi=300)

    return cluster_metrics_dict
