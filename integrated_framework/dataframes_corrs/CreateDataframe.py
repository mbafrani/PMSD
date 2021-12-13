import pandas as pd
import networkx as nx



def create_dataframes_from_equations(equation_set, name_space):
    """
    function used to create data frames to keep the raw information about discovered equations. Here, we only filter
    the very tiny coefficients that may happen in some equations found by a polynomial regression e.g., remove factor
    with a coefficient = 3.123532 * e-10 (this case can happen when the corresponding equation discovered by
    polynomial regression)

    :param name_space: a dictionary contains all pair-wise name relationship, e.g., Arrival rate1D : arrival_rate1d
    :param equation_set: the discovered equation sets without tiny coefficient (i,e, coefficients is basically 0 )
    :return:
        from discovered equation set, generate dataframe contain equation information
        raw_info_df: the data frame contains all raw information about discovered equations
        - only remove very tiny coefficient,

        raw_cause_effect_df: the data frame contains all raw informtion about discovered equations, instead of keep the
        value of exact coefficient, but just use '+' and '-' to express positive and negative influence
        - still, only remove very tiny coefficient

        all_lines: a set object keep all lines information (only remove very tiny coefficient)
    """

    # parameters = list(sd_log.columns)
    # e.g.,  {arrival_rate1d: Arrival rate1D}

    reversed_name_space = {v: k for k, v in name_space.items()}
    feature_list = name_space.keys()

    # create a df to maintain all raw information about discovered equations
    raw_info_df = pd.DataFrame(
        0,
        index=pd.Index(
            feature_list,
            name='Variables'),
        columns=feature_list, dtype=float)
    # raw_info_df['Intercept'] = float(0)

    # create a df to maintain the cause and effect relation between parameters
    raw_cause_effect_df = pd.DataFrame(
        '#',
        index=pd.Index(
            feature_list,
            name='Variables'),
        columns=feature_list, dtype=str)

    # create a set to maintain all line information to prepare for building a CLD
    all_lines = set()

    for d_para, info in equation_set.items():
        # the last element in the value part is the intercept and model name,
        # skip it
        for coef, ind_para in info[:-1]:

            if isinstance(
                    ind_para,
                    str):   # e.g., (0.05145244415262149, 'finish_rate8h')
                raw_info_df.at[reversed_name_space[d_para],
                       reversed_name_space[ind_para]] = round(coef,5)
                all_lines.add((reversed_name_space[ind_para], reversed_name_space[d_para]))

                if coef > 0:
                    raw_cause_effect_df.at[reversed_name_space[d_para],
                       reversed_name_space[ind_para]] = '+'
                else:
                    raw_cause_effect_df.at[reversed_name_space[d_para],
                                           reversed_name_space[ind_para]] = '-'
            else:
                if len(ind_para) == 1 and ind_para[0][1] == 1:  # e.g., (-0.0027121246389220743, [('num_in_process_case8h', 1)])
                    raw_info_df.at[reversed_name_space[d_para],
                           reversed_name_space[ind_para[0][0]]] = round(coef,5)
                    all_lines.add((reversed_name_space[ind_para[0][0]], reversed_name_space[d_para]))
                    if coef > 0:
                        raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[ind_para[0][0]]] = '+'
                    else:
                        raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[ind_para[0][0]]] = '-'
                elif len(ind_para) == 1 and ind_para[0][1] != 1:
                    s = reversed_name_space[ind_para[0][0]] + '^{}'.format(ind_para[0][1])
                    raw_info_df[s] = float(0)
                    raw_info_df.at[reversed_name_space[d_para], s] = round(coef,5)
                    all_lines.add((reversed_name_space[ind_para[0][0]], reversed_name_space[d_para]))
                    if raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[ind_para[0][0]]] == '#':
                        if coef > 0:
                            raw_cause_effect_df.at[
                                reversed_name_space[d_para], reversed_name_space[ind_para[0][0]]] = '+'
                        else:
                            raw_cause_effect_df.at[
                                reversed_name_space[d_para], reversed_name_space[ind_para[0][0]]] = '-'
                else:
                    temp = []
                    for pa, order in ind_para:
                        if order == 1:
                            temp.append(reversed_name_space[pa])
                        else:
                            temp.append('{}^{}'.format(reversed_name_space[pa], order))
                        all_lines.add((pa, d_para))
                        if raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[pa]] == '#':
                            if coef > 0:
                                raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[pa]] = '+'
                            else:
                                raw_cause_effect_df.at[reversed_name_space[d_para], reversed_name_space[pa]] = '-'
                    label = '*'.join(temp)
                    raw_info_df[label] = float(0)
                    raw_info_df.at[reversed_name_space[d_para], label] = round(coef,5)
        # raw_info_df.at[reversed_name_space[d_para],'Intercept'] = info[-1][0]
    # raw_info_df.to_csv('raw_dataframe', index=False)
    return raw_info_df, raw_cause_effect_df, all_lines


def simplify_df(dataframe):
    # simplify the original dataframe to make the structure of generated CLD clean
    """
    simplify the original data frame to make the structure of generated CLD clean
    three main tasks here:
    first， remove factors with small coefficient e.g., y = 5*a + 0.05*b, then factor b should be removed
    second, break cycles
    third, optimize the structure by remove chain effect e.g., if a -> b ->c then remove  a -> c


    :param dataframe: the data frame contains all raw information about discovered equations
    :return: the simplified data frame contains essential equation information, and also the corresponding simplified
    causal-effect data frame
    """
    new_df = dataframe.copy(deep=True)

    labels = list(new_df.columns)
    removed_pair = set()
    for index, row in new_df.iterrows():
        values = list(row)
        max_value = abs(max(values))
        # print([abs(i) / max_value for i in row])
        if max_value !=0:
            for cur_para in labels:
                if abs(row[cur_para]) / max_value < 0.05:
                    removed_pair.add((cur_para, index))

    for s, e in removed_pair:
        new_df.at[e,s] = 0

    # drop trivial columns which only contains 0
    removed_label =[]
    for para in labels:
        if new_df[para].isin([0]).all() and para not in new_df.index:
            removed_label.append(para)
    new_df.drop(removed_label, inplace=True, axis=1)

    # now we can check if there are some cycles in the lines
    all_lines = set()
    all_lines_without_coef = set()
    remain_labels = new_df.columns
    for index, row in new_df.iterrows():
        for rl in remain_labels:
            if row[rl] !=0:
                all_lines.add((rl, index, row[rl]))
                all_lines_without_coef.add((rl, index))

    # check cycles
    from equation_processing import ProcessEquations

    edges_2b_re_moved = ProcessEquations.edges_2be_removed(all_lines)

    remain_lines = [(s, e) for s, e in all_lines_without_coef if (s, e) not in edges_2b_re_moved]

    #  remove chain effect
    #redundant_lines = remove_chain_effect(remain_lines)
    #for s, e in redundant_lines:
       #edges_2b_re_moved.add((s,e))

    # update current data frame
    for s, e in edges_2b_re_moved:
       new_df.at[e, s] = 0

    # create a df store causal effect, i.e., positive(+) and negative(-) relation
    causal_df = pd.DataFrame(
        '#',
        index=pd.Index(
            remain_labels,
            name='Variables'),
        columns=remain_labels, dtype=str)

    for index, row in new_df.iterrows():
        for rl in remain_labels:
            if row[rl] == 0:
                causal_df.at[index, rl] = '#'
            elif row[rl] > 0 :
                causal_df.at[index, rl] = '+'
            else:
                causal_df.at[index, rl] = '-'

    return new_df, causal_df


def remove_chain_effect(lines):
    """
    break chain effect e.g., if we have A -> B -> C, then we should remove (A, C) from the given lines [(A,C), (A,B),
    (B, C)] because of we assume the cause effect from A to C can be represented by the factor B

    use a linked list to check the chain effect, start with the node only have out-bound links (if a node has some
    in-bound links, then it definitely on one longer chain)

    i.e., this is a transitive graph reduction problem, please refer to the following link
    https://en.wikipedia.org/wiki/Transitive_reduction

    :param lines: lines that from a DAG
    :return: the 'redundant line'
    """

    dg = nx.DiGraph(lines)
    tr = nx.transitive_reduction(dg)
    removed_lines = [(s,e) for (s, e) in lines if (s, e) not in list(tr.edges)]

    return removed_lines
