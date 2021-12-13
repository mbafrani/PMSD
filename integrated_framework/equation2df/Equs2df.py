import pandas as pd
# from integrated_framework.equation_processing import useless


def lines_to_df(name_space, stock_lines, big_coef_lines, small_coef_lines,  removed_lines):
    """

    :param name_space: a dictionary contains all pair-wise name relationship, e.g., Arrival rate1D : arrival_rate1d
    :param stock_lines: list contains all stock line information
    :param big_coef_lines:   list contains lines which have a coefficient greater or equal than a threshold
    :param small_coef_lines: list contains lines which have a coefficient smaller than a threshold
    :param removed_lines:    list contians all lines need to be removed



    -------
    :return: two dataframe, one is before further processing (stock_lines + big_coef_lines)
    one is after processing  (stock_lines + big_coef_lines) - (smaller_coef_lines + removed_lines)

    and the the value of dataframe at (i, j) means the "weight" of the line from node i to node j
    not actual coefficient because if we have a line  y = a* x_1 * x_2
    we give x_1 and x_2 same coefficient, i.e., (x_1, y, a) and  (x_2, y, a)
    so we don't know if we want to save the acutal coefficient, how much value should we put for x_1 and x_2
    but since there must be lines form x_1 and x_2 to y, so we give the "weight" as 1 if a > 0 and -1 if a <0
    """

    reversed_name_space = {v:k for k, v in name_space.items()}
    feature_list = name_space.keys()

    df_before_processing = pd.DataFrame(
        0,
        index=pd.Index(
            feature_list,
            name='Variables'),
        columns=feature_list)

    equations_lines = big_coef_lines | small_coef_lines
    lines_in_cld_full = stock_lines | equations_lines

    for start, end, coef in lines_in_cld_full:
        if coef > 0:
            df_before_processing.at[reversed_name_space[start], reversed_name_space[end]] = 1

        else:
            df_before_processing.at[reversed_name_space[start], reversed_name_space[end]] = -1

    df_after_processing = df_before_processing.copy(deep=True)

    # lines_not_in_cld_full = small_coef_lines | removed_lines
    for start, end, _ in small_coef_lines:
        df_after_processing.at[reversed_name_space[start], reversed_name_space[end]] = 0
    for start, end in removed_lines:
        df_after_processing.at[reversed_name_space[start], reversed_name_space[end]] = 0
    return df_before_processing, df_after_processing


def equs_to_df(sd_log, equations_before, equations_after):
    """
    convert equations information to dataframe

    :param sd_log: the input SD_Log
    :param equations_before: equation information before pre-processing
    :param equations_after:  equation information after pre-processing
    :return: df1 <--- the converted dataframe contains equations before pre-processing
             df2 <--- the converted dataframe contains equations after pre-processing

    """
    feature_list = sd_log.columns

    df1 = pd.DataFrame(
        0,
        index=pd.Index(
            feature_list,
            name='Variables'),
        columns=feature_list)
    df2 = df1.copy(deep=True)
    for p_var, (coef, _, e_var, _) in equations_before.items():
        df1.at[p_var, e_var] = coef

    for p_var, info in equations_after.items():
        for coef, e_var in info[:-1]:
            df2.at[p_var, e_var] = coef
    return df1, df2

