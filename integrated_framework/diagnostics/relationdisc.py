"""
- whole clustering of multivariate using k-means and dtw
- granger causation matrix between all features -> Relation Direction Detection (linear and non-linear)
- pearson correlation between all features -> Relation Type Detection (only linear)
- distance correlation between all features -> Relation Type Detection (linear and non-linear)

- causation matrix between two SDLogs
- correlation between two SDLogs
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
#import nonlincausality as nlc
import dcor


def dtw_visual(x, y):  # shape X, Y np.array([0., 0, 1, 2, 1, 0, 2, 1, 0, 0])
    """
    Plot to show how dtw works
    :param x: time series
    :param y: time series
    :return:
    """
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    path = dtw.warping_path(x, y)
    dtwvis.plot_warping(x, y, path, filename="tmp.png")


def k_means_clustering(sd_log):
    """
    k_means clustering of all features using dtw for multivariate time series
    :param sd_log: sd_log object
    :return: cluster_metrics_dict: dict with clusters as key and features as values
    """
    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
    from tslearn.utils import to_time_series_dataset
    from tslearn.preprocessing import TimeSeriesScalerMinMax

    data = sd_log.data
    # TODO handle outliers
    tmp = sd_log.waiting_time
    data.drop(columns=[sd_log.waiting_time], inplace=True)
    X = []
    # Get data as numpy array
    for col in data.columns:
        X.append(sd_log.get_points(col))

    # Normalize the data (y = (x - min) / (max - min))
    data_norm = data.copy()
    for column in data_norm.columns:
        data_norm[column] = (data_norm[column] - data_norm[column].min()) / (
                data_norm[column].max() - data_norm[column].min())

    X = TimeSeriesScalerMinMax().fit_transform(X)
    X = to_time_series_dataset(X)

    #  Find optimal # clusters by
    #  looping through different configurations for # of clusters and store the respective values for silhouette:
    sil_scores = {}
    for n in range(2, len(data.columns)):
        model_tst = TimeSeriesKMeans(n_clusters=n, metric="dtw", n_init=10)
        model_tst.fit(X)
        sil_scores[n] = (silhouette_score(X, model_tst.predict(X), metric="dtw"))

    opt_k = max(sil_scores, key=sil_scores.get)
    model = TimeSeriesKMeans(n_clusters=opt_k, metric="dtw", n_init=10)
    labels = model.fit_predict(X)
    print(labels)

    # build helper df to map metrics to their cluster labels
    df_cluster = pd.DataFrame(list(zip(data.columns, model.labels_)), columns=['metric', 'cluster'])

    # make some helper dictionaries and lists
    cluster_metrics_dict = df_cluster.groupby(['cluster'])['metric'].apply(lambda x: [x for x in x]).to_dict()
    cluster_len_dict = df_cluster['cluster'].value_counts().to_dict()
    clusters_dropped = [cluster for cluster in cluster_len_dict if cluster_len_dict[cluster] == 1]
    clusters_final = [cluster for cluster in cluster_len_dict if cluster_len_dict[cluster] > 1]

    print('Plotting Clusters')

    fig, axs = plt.subplots(opt_k)  # , figsize=(10, 5))
    # fig.suptitle('Clusters')
    row_i = 0
    # column_j = 0
    # For each label there is,
    # plots every series with that label
    for cluster in cluster_metrics_dict:
        for feat in cluster_metrics_dict[cluster]:
            axs[row_i].plot(data_norm[feat], label=feat, alpha=0.4)
            axs[row_i].legend(loc="best")
        if len(cluster_metrics_dict[cluster]) > 100:
            # TODO draw mean in red if more than one cluster
            tmp = np.nanmean(np.vstack(cluster), axis=1)
            axs[row_i].plot(tmp, c="red")
        axs[row_i].set_title("Cluster " + str(cluster))
        row_i += 1
        # column_j += 1
        # if column_j % k == 0:
        #    row_i += 1
        #    column_j = 0
    plt.show()

    # return dict {cluster_id: features}
    return cluster_metrics_dict


def grangers_causation_matrix(sd_log, test='ssr_ftest', plot=False, save_hm=False, outputpath=None, verbose=False, maxlag=6):
    """
    Check Granger Causality of all possible combinations of the Time series. The values in the table
    are the P-Values.
    If p_value < 0.05 then the corresponding X causes the Y and the relation is saved in the relations array
    with the desired lag: [(t2,t1)]: #lag -> t2 is caused by t1 in #lag

    sd_log      : sd_log object
    """
    if sd_log.isStationary:
        data = sd_log.data
    else:
        data, n_diff = sd_log.data_diff
    relations = {}
    exogenous_factors = []
    # TODO constant colums lead to error
    # data.drop(columns=[sd_log.waiting_time], inplace=True)
    #  drop constant columns
    data = data.loc[:, (data != data.iloc[0]).any()]
    variables = data.columns
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for t1 in df.columns:
        for t2 in df.index:
            test_result = grangercausalitytests(data[[t2, t1]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {t2}, X = {t1}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            lag = p_values.index(min_p_value) + 1
            df.loc[t2, t1] = min_p_value
            if min_p_value < 0.05:
                relations[(t2, t1)] = lag
            else:
                exogenous_factors.append((t2, t1))

    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    print(relations)
    if plot:
        plot_heatmap(data=df, title="Grangers Causality Among Features", save_plot=save_hm, outputpath=outputpath)
    return df, relations, exogenous_factors


def non_linear_granger_causation(sd_log, sd_log2=None, verbose=False, plot=False, save_hm=False, outputpath=None, maxlag=6):
    if sd_log.isStationary:
        data_one = sd_log.data
    else:
        data_one, n_diff_d1 = sd_log.data_diff
    if sd_log2:
        if sd_log2.isStationary:
            data_two = sd_log2.data
        else:
            data_two, n_diff_d2 = sd_log2.data_diff
        data_two = data_two.loc[:, (data_two != data_two.iloc[0]).any()]

    relations = {}
    exogenous_factors = []
    # TODO constant colums lead to error
    # data.drop(columns=[sd_log.waiting_time], inplace=True)
    #  drop constant columns
    data_one = data_one.loc[:, (data_one != data_one.iloc[0]).any()]
    variables = data_one.columns
    if sd_log2:
        variables2 = data_two.columns
        df = pd.DataFrame(np.zeros((len(variables2), len(variables))), columns=variables, index=variables2)
        data = pd.concat([data_one, data_two], axis=1).fillna(0)
    else:
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        data = data_one

    for t1 in df.columns:
        for t2 in df.index:
            results_ARIMAX = 0#nlc.nonlincausalityARIMAX(data[[t2, t1]].to_numpy(), d=0, maxlag=maxlag, plot=False)
            # results_NN = nlc.nonlincausalityNN(data[[t2, t1]].to_numpy(), maxlag=maxlag, NN_config=['d', 'dr', 'd', 'dr'],
            #                                    NN_neurons=[250, 0.1, 100, 0.1], run=10, epochs_num=[50, 50],
            #                                    learning_rate=[0.001, 0.0001], batch_size_num=64, plot=True,
            #                                    verbose=False)
            p_values = [round(results_ARIMAX[i + 1][0][0]['Wilcoxon test'][0][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {t2}, X = {t1}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            lag = p_values.index(min_p_value) + 1
            df.loc[t2, t1] = min_p_value
            if min_p_value < 0.05:
                relations[(t2, t1)] = lag
            else:
                exogenous_factors.append((t2, t1))

    df.columns = [var + '_x' for var in variables]
    if sd_log2:
        df.index = [var + '_y' for var in variables2]
    else:
        df.index = [var + '_y' for var in variables]
    print(relations)
    if plot:
        title = "Nonlinear (NN) Grangers Causality Among Features"
        if sd_log2:
            title = "Nonlinear (NN) Grangers Causality Among two SDLogs"
        plot_heatmap(data=df, title=title,
                     save_plot=save_hm, outputpath=outputpath)
    return df, relations, exogenous_factors


def grangers_causation_matrix_2sdlogs(sd_log_one, sd_log_two, test='ssr_ftest', plot=False, save_hm=False, outputpath=None, verbose=False, maxlag=6):
    """
    Check Granger Causality of two SDLogs (sd_log_one causes sd_log_two)
    If p_value < 0.05 then the corresponding X causes the Y and the relation is saved in the relations array
    with the desired lag: [(t2,t1)]: #lag -> t2 is caused by t1 in #lag

    sd_log      : sd_log object
    """
    if sd_log_one.isStationary:
        data_one = sd_log_one.data
    else:
        data_one, n_diff_one = sd_log_one.data_diff

    if sd_log_two.isStationary:
        data_two = sd_log_two.data
    else:
        data_two, n_diff_two = sd_log_two.data_diff

    data_one = data_one.loc[:, (data_one != data_one.iloc[0]).any()]
    data_two = data_two.loc[:, (data_two != data_two.iloc[0]).any()]
    data = pd.concat([data_one, data_two], axis=1).fillna(0)
    relations = {}
    exogenous_factors = []
    # TODO constant colums lead to error
    # data.drop(columns=[sd_log.waiting_time], inplace=True)
    #  drop constant columns
    data = data.loc[:, (data != data.iloc[0]).any()]
    variables_one = data_one.columns
    variables_two = data_two.columns
    df = pd.DataFrame(np.zeros((len(variables_one), len(variables_two))), columns=variables_two, index=variables_one)
    for t1 in df.columns:
        for t2 in df.index:
            test_result = grangercausalitytests(data[[t2, t1]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {t2}, X = {t1}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            lag = p_values.index(min_p_value) + 1
            df.loc[t2, t1] = min_p_value
            if min_p_value < 0.05:
                relations[(t2, t1)] = lag
            else:
                exogenous_factors.append((t2, t1))

    df.columns = [var + '_x' for var in variables_two]
    df.index = [var + '_y' for var in variables_one]
    print(relations)
    if plot:
        plot_heatmap(data=df, title="Grangers Causality Among two SDLogs", save_plot=save_hm, outputpath=outputpath)
    return df, relations, exogenous_factors


def corr_pearson(sd_log, sd_log2=None, plot=False, save_hm=False, outputpath=None):
    """
    Plots a heatmap with values as pearson correlation among all features for linear relation type detection
    @param sd_log2: second sd_log object - only if you would like to find relations between two logs i.e. activities
    @param sd_log: sd_log object
    @param outputpath:
    @param save_hm:
    @param plot:
    :return: return df.corr() of the sd_log
    """
    data1 = sd_log.data
    #  drop constant columns
    data1 = data1.loc[:, (data1 != data1.iloc[0]).any()]
    if sd_log2:
        data2 = sd_log2.data.loc[:, (sd_log2.data != sd_log2.data.iloc[0]).any()]
        data = pd.concat([data1, data2], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']
        title = "Pearson's Correlation Among two SDLogs"
    else:
        data = data1.corr()
        title = "Pearson's Correlation Among Features"
    if plot:
        plot_heatmap(data=data, title=title, save_plot=save_hm, outputpath=outputpath)
    return data


def corr_distance(sd_log, plot=False, save_hm=False, outputpath=None):
    # long runtime
    #import dcor
    data = sd_log.data
    #  drop constant columns
    data = data.loc[:, (data != data.iloc[0]).any()]
    feat_names = data.columns.tolist()
    df_dcor = pd.DataFrame(index=feat_names, columns=feat_names)

    k = 0
    for feat_i in feat_names:
        tmp = data.loc[:, feat_i]
        v1 = data.loc[:, feat_i].to_numpy()

        for feat_j in feat_names[k:]:
            v2 = data.loc[:, feat_j].to_numpy()

            rez = dcor.distance_correlation(v1, v2)

            df_dcor.loc[feat_i, feat_j] = float(rez)
            df_dcor.loc[feat_j, feat_i] = float(rez)

        k += 1
    # plot as heatmap
    if plot:
        plot_heatmap(df_dcor.astype(float), title="Distance Correlation Among Features",
                     save_plot=save_hm, outputpath=outputpath)
    return df_dcor


def corr_distance_2sdLogs(sd_log1, sd_log2, plot=False, save_hm=False, outputpath=None):
    # long runtime
    data1 = sd_log1.data
    data2 = sd_log2.data
    #  drop constant columns
    data1 = data1.loc[:, (data1 != data1.iloc[0]).any()]
    data2 = data2.loc[:, (data2 != data2.iloc[0]).any()]

    feat_names1 = data1.columns.tolist()
    feat_names2 = data1.columns.tolist()
    df_dcor = pd.DataFrame(index=feat_names1, columns=feat_names2)

    for feat_i in feat_names1:
        tmp = data1.loc[:, feat_i]
        v1 = data1.loc[:, feat_i].to_numpy()

        for feat_j in feat_names2:
            v2 = data2.loc[:, feat_j].to_numpy()

            rez = dcor.distance_correlation(v1, v2)

            df_dcor.loc[feat_i, feat_j] = float(rez)

    # plot as heatmap
    if plot:
        plot_heatmap(df_dcor.astype(float), title="Distance Correlation Among two SDLogs",
                     save_plot=save_hm, outputpath=outputpath)
    return df_dcor


def plot_heatmap(data, title, show_plot=True, save_plot=False, outputpath=None):
    # plot as heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    sns_hm =sns.heatmap(data,
                        cmap=sns.diverging_palette(220, 20, as_cmap=True),
                        square=True,
                        cbar_kws={'shrink': .9},
                        ax=ax,
                        annot=True,
                        linewidths=0.1, vmax=1.0, linecolor='white',
                        annot_kws={'fontsize': 12})
    plt.title(title)
    if show_plot:
        plt.show()
    if save_plot:
        sns_hm.get_figure().savefig(outputpath)
