import pandas as pd
import dcor
import collections
import warnings
warnings.filterwarnings("ignore")


def df_derived_by_shift(df, lag=0, non_der=[]):
    """
    create the dataframe with lag
    :param df: the input sd_log
    :param lag: a integer value, indicates the lag
    :param non_der: parameters that will exclude from lagged-log creation, set default as []
    :return: a lagged-log
    """

    temp = df.copy(deep=True)
    if not lag:
        return temp

    cols = collections.defaultdict(list)

    for i in range(0, lag + 1):
        for para in list(temp.columns):
            if para not in non_der:
                cols[para].append('{}_{}'.format(para, i))

    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=temp.index)

        i = 1
        for c in columns:
            dfn[c] = temp[k].shift(periods=i)
            i += 1
        temp = pd.concat([temp, dfn], axis=1, join='outer')
    return temp


def check_correlation(sd_log):
    """

    :param sd_log: the input sd_log
    :return: the distance correlation and lag correlation
    """

    feature_list = sd_log.columns

    # create lag log
    lag_sd_log = df_derived_by_shift(sd_log, 7)   # set the maximum lag as 7
    # check lag correlation
    all_correlations = lag_sd_log.corr(method=calcualte_dcc).loc[feature_list]

    dcc_correlations = all_correlations[feature_list]
    lag_correlations = all_correlations.iloc[:, len(feature_list):]

    return dcc_correlations, lag_correlations


def calcualte_dcc(x, y):
    """
    call function that will be used during checking the lag correlation
    """
    return dcor.distance_correlation(x, y)


def pick_strong_correlations(dcc, lag):
    """

    :param dcc: the dataframe contains distance correlation information
    :param lag: the dataframe contains lag correlation information
    :return: a dataframe indicates with which parameter, the variable Var has the highest correlation value. ROW index
    indicates the variable Var, and column index indicates the pair-wise parameter, content in the cell is the variable
    with which 'Var' has the highest correlations value
    """
    """
    for idx, row in lag.iterrows():
        for r in row.index:

            if r.split('_')[-1] == '0':
                r1 = '_'.join(r.split('_')[0:-1])
                print(r1)
                print()
                lag.at[idx, r] = dcc.at[idx, r1]
    """
    features = list(dcc.columns)
    # lag_features = list(lag.columns)
    output = pd.DataFrame(
        '#',
        index=pd.Index(
            features,
            name='Variables'),
        columns=features,
        dtype=str)

    for idx, row in output.iterrows():
        for para in features:
            # add the distance correlation coefficient
            temp = [(idx, para, dcc.loc[idx][para])]
            for n in range(1, 8):  # remember the maximum lag is 7
                cur = para + '_{}'.format(n)
                temp.append((idx, cur, lag.loc[idx][cur]))
            temp.sort(key=lambda x: x[-1], reverse=True)

            if temp[0][1] in features and temp[0][1] != idx:
                if type(temp[0][1][-1])!=int:
                    output.at[idx, para] = 0    # without lag
                else:
                    output.at[idx,para]=int(temp[1][1][-1])
            elif temp[0][1] in features and temp[0][1] == idx:
                #output.at[idx, para] = '#'   # auto correlation
                output.at[idx, idx] = int(temp[1][1][-1])
            else:
                output.at[idx, para] = int(temp[0][1][-1])  # with lag

    return output

def pick_strong_corr(dcc,lag):
    features = list(dcc.columns)
    # lag_features = list(lag.columns)
    output1 = pd.DataFrame(
        '#',
        index=pd.Index(
            features,
            name='Variables'),
        columns=features,
        dtype=str)
    for idx, row in output1.iterrows():
        for para in features:
            # add the distance correlation coefficient
            temp = [(idx, para, dcc.loc[idx][para])]
            for n in range(1, 8):  # remember the maximum lag is 7
                cur = para + '_{}'.format(n)
                temp.append((idx, cur, lag.loc[idx][cur]))
            temp.sort(key=lambda x: x[-1], reverse=True)
            if temp[0][1] in features and temp[0][1] != idx:

                output1.at[idx, para] = (temp[0][2])
                #output1.at[idx, para] = 0  # without lag
            elif temp[0][1] in features and temp[0][1] == idx:
                # output.at[idx, para] = '#'   # auto correlation
                output1.at[idx, idx] = (temp[1][2])
            else:
                output1.at[idx, para] = (temp[0][2])  # with lag
    return output1

def check_real_corr(output_df,output_df1,data):
    normal_corr = data.corr()
    for idx, row in output_df.iterrows():
        for r in row.index:
            if output_df1.at[idx, r] == 0 or abs(output_df1.at[idx, r]) <= 0.4:
                output_df.at[idx, r] = 0
            else:
                output_df.at[idx, r] = output_df.at[idx, r] + 1
            if normal_corr.at[idx, r] < 0 :
                output_df.at[idx, r] = output_df.at[idx, r] * -1
    return output_df