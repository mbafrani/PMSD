import collections
import networkx as nx


def pre_processing_equations(mean_values_of_features, all_best_equations):
    """
    do some pre-processing works to eliminate feature coefficient with round(coefficient,3) == 0   round(0.0001,3) = 0
    these cases happen with a polynomial model often, so we decide to remove them, and update the coefficient
    (basically +0, because these coefficient a too samll, e.g., -3.04942913e-15 )

    :param mean_values_of_features: is a dictionary contains all mean value of process variables in given SD-Log
    :param all_best_equations: is a dictionary {target_variables: [coef, intercept, IVs, name_of_selected_model]}
           target_variables and name of selected_model is string type, coef and IVs are list type (pairwise),
           intercept is a number

    ---------
    return a dictionary all_best_equations-without_tiny_coef,  without extremely small coefficient.
    in form of {feature : [(coef, var),...,(intercept, model_name)]}, if model is 'polynomial regression'
    then var is a list of variables with its order information


    """
    all_best_equations_without_tiny_coef = collections.defaultdict(list)

    for feature, equ_info in all_best_equations.items():
        step = 0  # intercept increase value
        coefs, intercept, IVs, name_of_model = equ_info[0], equ_info[1], equ_info[2], equ_info[3]

        zipped_coef_ivs = zip(coefs, IVs)

        if name_of_model == 'polynomial regression':
            # the element in zipped object becomes (coef, [(var, order)])
            # var is the explanatory variables in the equation, order is the
            # number of mathematical order
            for coef, vars_and_orders in zipped_coef_ivs:
                temp = coef  # intercept increase value
                if round(coef, 3) == 0:
                    for var, order in vars_and_orders:
                        temp *= pow(mean_values_of_features[var], order)
                    step += temp
                else:
                    all_best_equations_without_tiny_coef[feature].append(
                        (coef, vars_and_orders))
        else:
            for coef, var in zipped_coef_ivs:
                if round(coef, 3) == 0:
                    step += (mean_values_of_features[var] * coef)
                else:
                    all_best_equations_without_tiny_coef[feature].append(
                        (coef, var))
        intercept += step   # update model intercept
        all_best_equations_without_tiny_coef[feature].append(
            (intercept, name_of_model))  # add intercept and model name
    return all_best_equations_without_tiny_coef


def process_equations_new(
        sd_log,
        best_equations_without_tiny_coefficient,
        sf_vars,
        data_vars=None,
        threshold=0.005):
    """

    :param sd_log: the input SD_Log, dataframe
    :param best_equations_without_tiny_coefficient: the equation set without very tiny coefficient (< 0.001)
        a dictionary, in form {process variable: [(coef, var)... (intercept, name of model)]}
        if name_of_model == 'polynomial regression', var is a list of tuples: [(var, order)]
    :param sf_vars: stock and flow variables. A dictionary in form {stock variable: [inflows, outflows]}, inflow and
    outflows are lists contain flow variables.
    :param data_vars: the data variables. A list contain all data variables. if no data variables specified, the
    framework will detect the node without in-bound links as data variables.
    :param threshold: a predefined threshold to filter less significant variables out to simplify the structure of
    CLD and SDF

    ---------
    return the processed equations and all distinct line in given equations

    lines_with_bigger_coefficient is a set object contains all lines information in form of (from, to, coefficient)
    and the coefficient greater or equal than the pre-defined threshold
    for lines in a polynomial model, we give all explanatory variables the same coefficient
    for example, y = a * x_1 * x_2 + c, we add line(x_1, y, a) and line(x_2, y, a)

    lines_with_samller_coefficient is a set object contains all lines information in form of (from, to, coefficient)\
    and the coefficient smaller than the pre-defined threshold

    """

    # this the lines information, in which all lines have a coefficient value
    # great or equal than the threshold
    lines_with_bigger_coefficient = set()

    # this is the lines information, in which all lines have a coefficient
    # value less than the threshold
    lines_with_smaller_coefficient = set()

    # stock_lines is set object contains all lines information from inflow and
    # outflow to stock variables
    stock_lines = set()

    if not best_equations_without_tiny_coefficient:
        return 'The best equation set seems empty, please re-run the framework and discover best equations first'

    processed_equations = collections.defaultdict(list)

    # add stock variable information, since stock variables can only be influenced by flows variables
    # so we fix the form of stock variables as: stock_variable =
    # inflow_variable - outflow_variable + constant_value

    for stock, flows in sf_vars.items():
        inflows, outflows = flows[0], flows[1]
        if inflows:
            for inflow in inflows:
                processed_equations[stock].append(
                    (1, inflow))   # inflow increase the value of stock
                # add inflow lines (from, to, coefficient)
                stock_lines.add((inflow, stock, 1))
        if outflows:
            for outflow in outflows:
                processed_equations[stock].append(
                    (-1, outflow))  # outflow decrease the value of stock
                stock_lines.add(
                    (outflow, stock, -1))              # add outflow lines

        # add the initial value of stock variable, sd index start with 1
        processed_equations[stock].append([sd_log[stock][1], 'stock'])

    # add auxiliary variable information
    for arr, arr_equation_info in best_equations_without_tiny_coefficient.items():

        if arr in processed_equations:  # since we fix stocks variable already, so, we don't add the links and equation information
            continue

        if arr in (data_vars or []):   # if user specify data variables already, we also don't save data variables' link and equation information, because the equation for data variables are fixed too (value of data variables read from local)
            continue

        intercept, name_of_model = arr_equation_info[-1]
        coef_and_vars = arr_equation_info[:-1]

        step = 0  # this is the value need to be added to the intercept, because we didn't count this lines in the equations anymore

        for coef, variables_info in coef_and_vars:
            if abs(coef) >= threshold:  # if the absolute value of coefficient is big enough, we add it to the processed_equations and also the lines_with_bigger_coefficient set
                if isinstance(variables_info,
                              list):       # this is a polynomial model
                    for cur_var, var_order in variables_info:
                        # add line information to
                        # lines_with_bigger_coefficient, give all
                        lines_with_bigger_coefficient.add((cur_var, arr, coef))
                        # variables the same coefficient
                        # add lines information to processed_equations
                    # processed_equations[arr].append((coef, variables_info))
                else:
                    lines_with_bigger_coefficient.add(
                        (variables_info, arr, coef))
                processed_equations[arr].append(
                    [coef, variables_info])

            else:   # the coefficient less than the pre-defined threshold, add them to the lines_with smaller_coefficient set, and we calcualte a number 'step' which should be added to the intercept

                if isinstance(
                        variables_info,
                        list):  # polynomial model, variables has a polynomial order
                    for cur_var, var_order in variables_info:
                        lines_with_smaller_coefficient.add(
                            (cur_var, arr, coef))
                        step += pow(sd_log[cur_var].mean(), var_order) * coef
                else:
                    step += coef * sd_log[variables_info].mean()
                    lines_with_smaller_coefficient.add(
                        (variables_info, arr, coef))

        intercept += step
        processed_equations[arr].append(
            [intercept, name_of_model])   # update the intercept value

        # then we only use lines_with_bigger_coefficient to detect cycles
        # why we don't add lines in stock_line? because it's acceptable that
        # cycles happen between stock and inflow or outflow

        # stocks_lines + lines_with_bigger_coefficient + lines_with_smaller_coefficient should be used to create the
        # initial CLD diagram.

    return processed_equations, stock_lines, lines_with_bigger_coefficient, lines_with_smaller_coefficient


def detect_cycles(lines):
    """
    use lines information to create a directed graph, then detect cycles

    :param lines: a set object containing all lines information , {(from, to, coef)}, the input should be
    the lines_with_bigger_coefficient from process_equations_new
    :return: cycles, a list of lists containing cycles
    """

    g = nx.DiGraph()
    for s, e in lines:
        g.add_edge(s, e)

    cycles = [cycle for cycle in nx.simple_cycles(g)]
    return cycles


def edges_2be_removed(lines):
    """
    detect cycles and remove lines in cycles according to the node degree
    :param lines:
    :return:
    """

    removed_edges = set()

    # lines without coefficient
    lines_without_coef = set()
    for s, e, _ in lines:
        lines_without_coef.add((s, e))
    lines_without_coef = list(lines_without_coef)

    # count_outbound_links and inbnoud links for every node
    nodes = set([l[0] for l in lines])
    nodes_out = collections.defaultdict(list)
    nodes_in = collections.defaultdict(list)
    for node in nodes:
        for s, e, _ in lines:
            if s == node:
                nodes_out[node].append(e)
            if e == node:
                nodes_in[node].append(s)

    cycles = detect_cycles(lines_without_coef)

    while cycles:
        # sort by length, always prefer to handle the shortest cycle first
        cycles.sort(key=len, reverse=True)
        cur = cycles[-1]

        # if the cycle only contains two nodes, then the cycle is just a --> b
        # --> a
        if len(cur) == 2:
            cur.sort()
            out_num1, out_num2 = len(nodes_out[cur[0]]), len(nodes_out[cur[1]])
            # node cur[1] is more important
            if out_num1 < out_num2:
                # delete edge var1 ---> var2 : cur[0] --> cur[1]
                var1, var2 = cur[0], cur[1]
            # node cur[0] is more important
            elif out_num1 > out_num2:
                # delete edge var1 ---> var2 : cur[1] --> cur[0]
                var1, var2 = cur[1], cur[0]
            else:
                # if outbound number is equal, then compare inbound links
                in_num1, in_num2 = len(nodes_in[cur[0]]), len(nodes_in[cur[1]])
                # node cycles[1] is more important
                if in_num1 < in_num2:
                    # delete edge var1 ---> var2 : cur[0] --> cur[1]
                    var1, var2 = cur[0], cur[1]
                # node cycles[0] is more important:
                elif in_num1 > in_num2:
                    # delete  edge var1 --> var2 : cur[1] --> cur[0]
                    var1, var2 = cur[1], cur[0]
                else:                                    # two nodes have same outbound link and inbound link
                    # delete arbitrary link, here, we remove cur[0]-->cur[1]
                    var1, var2 = cur[0], cur[1]
            removed_edges.add((var1, var2))

            # remove from inbound and outbound set
            if var2 in nodes_out[var1]:
                nodes_out[var1].remove(var2)
            if var1 in nodes_in[var2]:
                nodes_in[var2].remove(var1)

            # remove from lines_without_coef set
            if (var1, var2) in lines_without_coef:
                lines_without_coef.remove((var1, var2))

        else:
            # more than two nodes form a cycle, we define the node importance by number of outbound edges
            # [a, b, c] : a--> b -->c-->a

            # add differ information from last node to first node
            adjacent_edges = [
                [cur[-1], cur[0], abs(len(nodes_out[cur[-1]]) - len(nodes_out[cur[0]]))]]

            for idx in range(1, len(cur)):
                adjacent_edges.append(
                    [cur[idx - 1], cur[idx], abs(len(nodes_out[cur[idx - 1]]) - len(nodes_out[cur[idx]]))])

            # sort by count difference
            adjacent_edges.sort(key=lambda x: x[-1])
            # select the biggest difference to remove
            removed_edges.add((adjacent_edges[-1][0], adjacent_edges[-1][1]))

            # remove from inbound and outbound set
            if adjacent_edges[-1][1] in nodes_out[adjacent_edges[-1][0]]:
                nodes_out[adjacent_edges[-1][0]].remove(adjacent_edges[-1][1])
            if adjacent_edges[-1][0] in nodes_in[adjacent_edges[-1][1]]:
                nodes_in[adjacent_edges[-1][1]].remove(adjacent_edges[-1][0])

            # remove from lines_without_coef set
            if (adjacent_edges[-1][0],
                    adjacent_edges[-1][1]) in lines_without_coef:

                lines_without_coef.remove(
                    (adjacent_edges[-1][0], adjacent_edges[-1][1]))
        cycles = detect_cycles(lines_without_coef)
    return removed_edges


def update_processed_equations(
        mean_values_sd,
        processed_equations,
        removed_edges):
    """
    since we get the removed_edges by detecting cycles, now we should remove this links from the processed_equations

    :param mean_values_sd  a dictionary contains all mean values of process variables in the given SD-Log
    :param processed_equations: a dictionary, process variables are keys and equations information (remove all links
    with coefficient smaller than a pre-defined threshold)
    {process_var: [[coef, explanatory variable], .., [intercept, name_of_model]])}
    if name_of_model == 'polynomial regression', then explanatory variables is a list of tuples [(var, poly_order)]
    if name_of_mode == 'stock', then process_var is a stock in SFD, equation is fixed: inflow - outflow + constant
    :param removed_edges: list of tuples in form of (start, end), contains all edges need to be removed from
    processed_equations

    -----------
    :return the updated_processed_equations  (this is the final equations information that will be used to construct
    the SFD and the CLD)
    - remove essential edges
    - update intercept
    """
    if not removed_edges:   # no edges need to be removed, just return the processed_equations
        return processed_equations

    for start, end in removed_edges:
        # get all equation information of node 'end' (a list object)
        all_edges_in_end = processed_equations[end]
        # print('{}'.format(end), all_edges_in_end)

        for idx, variables_info in enumerate(all_edges_in_end):

            # print('idx',idx)
            # print('info', variables_info)
            if isinstance(variables_info[1], list):
                # coef = all_edges_in_end[idx][0]
                vars_info = all_edges_in_end[idx][1]

                # replace the (start, end) edge in all_edges_in_end by
                # replacing the var_name (var_name == start) as its mean_value
                for jdx, vars_name_order in enumerate(vars_info):
                    var_name = vars_info[jdx][0]
                    var_order = vars_info[jdx][1]

                    if var_name == start:
                        # print('yesssss')
                        # add (mean_value, poly_order)
                        vars_info.append((mean_values_sd[var_name], var_order))
                        # remove(var, poly_order)
                        vars_info.remove((var_name, var_order))
            else:
                if start in variables_info:
                    # print('hereeeeee')
                    # list object, replace the variable by its mean_value
                    all_edges_in_end[idx][1] = mean_values_sd[start]
    # the returned equations is the one that we need to forward into a SFD
    # (cycle free but need to detect data variables first)
    return processed_equations


def detect_data_variables(
        namespace,
        processed_equations,
        var_fail_to_get_equations=None):
    """
    detect data variables
    if there is no links points to a variable, then this variables should be the data variable

    input is the
    :param processed_equations: the processed_equations (dictionary),key is process variable, value is the
    equation information (coef, var_name) normally var_name is a string type and indicates the name of a variable,
    but if this equation information come from a polynomial model, then var_name is a list of tuples,
    contain (variable_name, poly_order)

    and
    :param var_fail_to_get_equations: variables that are failed to get equations (but these variables are possible to
    shown in equations for other process variables, so we treat this kind of variables as data variables too)


    if we couldn't find a variable for selected process variable, then this process variable should be regarded as
    the data variable.
    In a SFD, there must be a variables whose value we already know (either as a constant value or value read directly
    from local) otherwise, there is still cycle to be found.

    """
    data_variables = set()

    for process_variables, variable_equation in processed_equations.items():
        count = 0
        for coef, vars_info in variable_equation[:-
                                                 1]:  # the last one is the (coef, name_of_model)
            if isinstance(vars_info, list):
                for var_name, _ in vars_info:
                    if isinstance(var_name, str):
                        count += 1
            else:
                if isinstance(vars_info, str):
                    count += 1
        if count == 0:
            data_variables.add(process_variables)
    data_variables = data_variables | var_fail_to_get_equations

    while not data_variables:
        print(
            'No data variable detected, we recommend to specify at least the inflow variable as data variable')
        print(list(namespace.values()))
        prompt = "\n Please select at least one data variable"
        data_variables = input(prompt)
        data_variables = [d.strip() for d in data_variables.split(',')]
        for d in data_variables:
            while d not in namespace.values():
                print(
                    'Your previous enter seems wrong, check it carefully and re-enter your selection')
                data_variables = None
    return data_variables


def rebuild_equations(processed_equations, data_variables):
    """
    build the equaitons in form of mathematical form : y = ax+b, so that, we can directly use it in the SFD
    and we skip data_variables, because we fetch value for data_variables from local

    ------
    return the mathematical form of processed equations, i.e., make them all like: y = ax+b
    """

    equations_in_mathematical = {}
    for process_variable, variable_equation in processed_equations.items():
        if process_variable not in data_variables:
            math_form = ''
            for coef, vars_info in variable_equation[:-1]:
                # add coefficient at the header   y = a ....
                sub_form = str(coef)
                if isinstance(vars_info, list):
                    for var_name, poly_order in vars_info:
                        # e.g., [0.333333333, [('arrival_rate1w', 1)]
                        if poly_order == 1:
                            sub_form += (' * ' + str(var_name))
                        else:  # [0.333333, ('service_time_per_case1w', 2)]
                            sub_form += ' * ' + \
                                '{}^{}'.format(var_name, poly_order)
                else:  # e.g., [-0.24433476, 'waiting_time_in_process_per_case1w']
                    sub_form += (' * ' + str(vars_info))
                math_form += (sub_form + ' + ')
            if round(variable_equation[-1][0], 3) == 0:
                math_form += '0'
            else:
                # add the intercept information
                math_form += str(variable_equation[-1][0])
            equations_in_mathematical[process_variable] = math_form

    # now we can forward this equations into a SFD
    return equations_in_mathematical
