""" all to introduce user interactions"""

def get_target_feature(data):
    prompt = "\n Please select one process feature as you want given feature set from your data, then press 'Enter' "
    print(list(data.columns))
    all_vars = set(data.columns)
    target_feature = input(prompt)
    while target_feature not in all_vars:
        prompt4 = "\n Your previous entered process variable seems doesn't exist in the given SD-Log, " \
                  "please enter the process variable carefully,then press 'Enter'."
        target_feature = input(prompt4)
    return target_feature


def drop_column_switcher(data):
    prompt = "\n Are there process features need to be dropped? If yes, enter the feature name and separated by commas in" \
             " case more than one and press 'Enter', otherwise just press 'Enter' to skip. "
    print(list(data.columns))
    all_vars = set(data.columns)
    dropped_feature = input(prompt).split(',')
    if dropped_feature == ['']:
        return None
    else:
        var_in_data = [
            True if var in all_vars else False for var in dropped_feature]
        # while True:
        while not all(var_in_data):
            prompt2 = "\n Your previous enter contains variables which are not included in given SD-Log, " \
                      "please try again carefully!"
            dropped_feature = input(prompt2).split(',')
            if dropped_feature == ['']:
                return None
            var_in_data = [
                True if var in all_vars else False for var in dropped_feature]
        return dropped_feature


def curve_fitting_switcher():
    curve_fitting_label = "\n Do you want to try with Curve Fitting on the given SD-Log, enter 'y' for 'Yes' and 'n' " \
                          "for 'No', then press 'Enter'"
    label = input(curve_fitting_label)
    while label not in ['Y', 'y', 'N', 'n']:
        promt = "\n Your input seems non-sense, please either enter 'Y'/'y' (stands for yes) or 'N'/'n (stands for no)"
        label = input(promt)
    return label


def poly_regression_switcher():
    poly_regression_label = "\n Do you want to try with polynomial regression on the given SD-Log, enter 'y' for 'Yes' " \
                            "and 'n' for 'No', then press 'Enter'"
    label = input(poly_regression_label)
    while label not in ['Y', 'y', 'N', 'n']:
        promt = "\n Your input seems non-sense, please either enter 'Y'/'y' (stands for yes) or 'N'/'n (stands for no)"
        label = input(promt)
    return label


def get_stock_and_flow_variables(sd_log):
    sf_variables = {}
    features = set(sd_log.columns)

    def get_input(features):
        # more_stock = True
        # while more_stock:
        print(list(sd_log.columns))

        prompt_stocks = "\n Please specify one stock variable at a time, without quotes,then press 'Enter'."
        prompt_inflows = "\n Please specify the corresponding inflow variables without quotes, separated by commas in " \
                         "case more than one inflows and press 'Enter'. If no inflow variables, just press 'Enter'to skip ."
        prompt_outflows = "\n Please specify the corresponding outflow variables without quotes, separated by commas in " \
                          "case more than one outflows and press 'Enter'. If no inflow variables, just press 'Enter' " \
                          "to skip."
        bad_input = "\n Your previous entered content seems doesn't exist in the given SD-Log, please enter " \
                    "the variable name carefully."
        stocks = input(prompt_stocks)

        while stocks not in features:
            stocks = input(bad_input)

        inflows = input(prompt_inflows)
        inflows = [f.strip() for f in inflows.split(',')]
        if inflows == ['']:
            inflows = None
            # sf_variables[stocks] = [None]
        else:
            input_inflow_in_sd = [True if f in features else False for f in inflows]
            while not all(input_inflow_in_sd):
                inflows = input(bad_input)
                inflows = [f.strip() for f in inflows.split(',')]
                input_inflow_in_sd = [True if f in features else False for f in inflows]

        outflows = input(prompt_outflows)
        outflows = [f.strip() for f in outflows.split(',')]

        if outflows == ['']:
            outflows = None
        else:
            input_outflow_in_sd = [True if f in features else False for f in outflows]

            while not all(input_outflow_in_sd):
                outflows = input(bad_input)
                outflows = [f.strip() for f in outflows.split(',')]
                input_outflow_in_sd = [True if f in features else False for f in outflows]
        return stocks, inflows, outflows

    stocks, inflows, outflows = get_input(features)

    intersections = None
    # one variable can't be inflow and outflow at the same time
    if inflows and outflows:
        intersections = [f for f in inflows if f in outflows]
    # there must have inflows and outflows
    elif not inflows and not outflows:
        while not inflows and not outflows:
            print('Stock variables must have at least one inflow or outflow variable, please re-enter!')
            stocks, inflows, outflows = get_input(features)
    while intersections:
        print('Variables can not be inflow and outflow variable at the same time, check your input carefully!')
        stocks, inflows, outflows = get_input(features)
        intersections = [f for f in inflows if f in outflows]
    # stocks, inflows, outflows = get_input(features)

    sf_variables[stocks] = [inflows, outflows]

    continue_prompt = "\n More stock and flow variables? press 'Y'/'y' to continue otherwise press 'N'/'n' to exit"
    label = input(continue_prompt)

    while True:
        while label not in ['y', 'Y', 'n', 'N']:
            print('Please type "y/Y" to continue or "n/N" to exit! Previous enter is nonsense.')
            label = input(continue_prompt)
        else:
            if label in ['y','Y']:
                stocks, inflows, outflows = get_input(features)
                sf_variables[stocks] = [inflows, outflows]
                label = input(continue_prompt)
            else:
                break
    return sf_variables



def get_stocks_and_flows(sd_log):
    goon = True
    sf_variables = []
    all_variables = set(sd_log.columns)
    print(list(sd_log.columns))
    # add dummy variable 'n' and 'N' to care cases if there is no inflow or outflow
    all_variables.add('n')
    all_variables.add('N')   # given stock variable.

    while goon:

        prompt1 = "\n Please specify the stock variable, without quotes,then press 'Enter'."
        prompt2 = "\n Please specify the corresponding flow variables without quotes, first inflow and then outflow" \
                  ", separated by commas. Use 'n' or 'N' to replace the corresponding spot in case there is no " \
                  "inflow or outflow variable, then press 'Enter'." \

        stocks = input(prompt1)
        while stocks not in all_variables:
            prompt4 = "\n Your previous entered stock variable seems doesn't exist in the given SD-Log, please enter " \
                      "the stock variable carefully, " \
                      "then press 'Enter'."
            stocks = input(prompt4)
        flows = input(prompt2)
        flows = [f.strip() for f in flows.split(',')]
        while len(flows) != 2:
            p = "\n Please enter the corresponding inflow variable and outflow variable, " \
                "previous enter seems not right"
            flows = input(p)
            flows = flows.strip().split(',')

        inflow, outflow = flows[0], flows[1]
        while (
            inflow not in all_variables) or (
            outflow not in all_variables) or (
            inflow in [
                'n',
                'N'] and outflow in [
                'n',
                'N']):
            prompt5 = "\n Your previous entered content seems wrong, please enter the corresponding stock variable " \
                      "carefully, separated by a single commas symbol and then press 'Enter'.".format(inflow, outflow)
            flows = input(prompt5)
            flows = [f.strip() for f in flows.split(',')]
            inflow, outflow = flows[0], flows[1]

            # print()
        sf_variables.append((stocks, flows))
        prompt3 = "\n More stock and flow variables? press 'Y'/'y' to continue otherwise press 'N'/'n' to exit"
        label = input(prompt3)
        if label in ['n', 'N']:
            goon = False
    return sf_variables


def get_data_variables(sd_log):
    all_variables = set(sd_log.columns)
    print(list(sd_log.columns))
    prompt1 = "\n Please specify the data variable(s) in case you know already, without quotes and separated by " \
              "commas if more than one and end with 'Enter'. In case you don't know which should be the data   " \
              "variables or just no data variables exist,just press 'Enter' "
    datas = input(prompt1)
    datas = [d.strip() for d in datas.split(',')]
    if datas == ['']:
        return None
    else:
        var_in_data = [
            True if var in all_variables else False for var in datas]
        # while True:
        while not all(var_in_data):
            prompt2 = "\n Your previous enter contains variables which are not included in given SD-Log, " \
                      "please try again carefully!"
            datas = input(prompt2)
            datas = [d.strip() for d in datas.split(',')]
            var_in_data = [
                True if var in all_variables else False for var in datas]
        return datas
