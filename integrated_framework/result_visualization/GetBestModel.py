import numpy as np


def select_best_model(training_res, mean_values, target_feature):
    """
    select the best model by comparing RMSE value / R^2 score on training/test data
    if RMSE value on training set and testing set are not similar, then this model may be overfitted
    same as the R^2 score. Moreover, if the R^2 score on training set is positive while on testing set
    is negative, then we may conclude that it's not a good choice to use this.

    Question: what's a good RMSE value ??
    -  there is no specific number to define a good RMSE value, we just try to minimize the value of RMSE.
    But if the RMSE is small on training and high on testing set, it's a indication of overfitting.
    So, we check whether the RMSE values vary on both data a lot. (We set a differ ratio as 20% ?)

    Another possibility is the RMSE value is big on both training and testing data : the value of RMSE are similar,
    but not good. ->> we don't take it, and recommend to do curve fitting (maybe a solution, because sometimes the data
    maybe too complex to express with just on model)

    Then the follow question is: how do we define a too bad RMSE value? -- depends on the scale of the process variable
    we set the differ ration as 1 * np.mean(process variable), i.e. if RMSE <= np.mean(process variable), we take it.


    Question: waht's a good R^2 value

    small R^2 doesn't always mean bad and chase high R^2 may also cause overfitting, we just set R^2 = 0.2
    -- if R^2 < 0.2 and RMSE is high ---> reject model
    -- if R^2_test and R^2_train differs a lot, e.g., R^2_train >0 and R^2 _test < 0: reject model

    """

    # rmse_list = [i[-1] for i in list(training_res.values())]
    # rmse_list is in form [[model, (rmse_test, rmse_train), (r2_test,
    # r2_train), rmse_test]]

    rmse_list = [[model, i[-3], i[-2], i[-1]]
                 for model, i in training_res.items()]
    # sort by the ascending order of RMSE_train value
    rmse_list.sort(key=lambda x: x[1][1], reverse=True)

    while rmse_list:
        cur = rmse_list.pop()
        rmse_test = round(cur[1][0], 3)
        rmse_train = round(cur[1][1], 3)
        cur_model = cur[0]

        # print('in the beginning the model is', cur_model)
        pivot = abs(mean_values[target_feature])
        # print('22222', (abs(rmse_train - min_rmse_test) / rmse_train))

        # (rmse_train > pivot) or
        if (cur_model == 'polynomial regression' and rmse_train != 0
            and (abs(rmse_train - rmse_test) / rmse_train)
                >= 0.05) or (cur_model != 'polynomial regression ' and rmse_train != 0 and
                             (abs(rmse_train - rmse_test) / rmse_train)
                             >= 0.5 and rmse_test >= 0.5 * pivot and rmse_train >= 0.5 * pivot) or (
                cur_model == 'polynomial regression' and rmse_train == 0 and
                rmse_test >= 0.05 * pivot) or (cur_model != 'polynomial regression' and rmse_train == 0 and
                                               rmse_test >= 0.5 * pivot):
            continue

        for model, summary in training_res.items():

            if summary[-1] == rmse_train and model != 'polynomial regression':

                # check r^2 score
                r2_test, r2_train = summary[-2][0], summary[-2][1]

                if (r2_train < 0.01 and r2_test * r2_train < 0) or (r2_train < 0.01 or r2_test <
                                                                    0.01):  # r2 score too small or r2_train r2_test differs a lot
                    continue
                else:
                    var_coef, intercept, feature_names = summary[0], summary[1], summary[2]
                    selected_training_method = model
                    return var_coef, intercept, feature_names, rmse_train, selected_training_method
            elif summary[-1] == rmse_train and model == 'polynomial regression':
                # check next model's rmse values
                next_min_rmse_train = rmse_list[-1][1][1]
                if abs(next_min_rmse_train - rmse_train) / rmse_train <= 0.2:
                    summary = training_res[rmse_list[-1][0]]
                    var_coef, intercept, feature_names = summary[0], summary[1], summary[2]
                    min_rmse = next_min_rmse_train
                    selected_training_method = rmse_list[1][0]
                    return var_coef, intercept, feature_names, min_rmse, selected_training_method

                else:
                    var_coef, intercept, feature_names = summary[0], summary[1], summary[2]
                    selected_training_method = model
                    return var_coef, intercept, feature_names, rmse_train, selected_training_method
    return None   # in this case, no best model is selected


def pretty_print_linear(
        target_feature,
        coefs,
        intercept=0,
        names=None,
        sort=False,
):
    """ print human readable results"""
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    equation = " + ".join("%s * %s" % (round(coef, 3), name)
                          for coef, name in lst if round(coef, 3) != 0)
    if intercept != 0:
        return target_feature + "=" + equation + \
            '%+.3f' % (round(intercept, 3))
    else:
        return target_feature + "=" + equation


def print_polynomail_equation(
        target_feature,
        coef,
        ivs,
        intercept=0
):
    lst = zip(coef, ivs)
    e, constant = '', 0
    for c, vars in lst:
        temp = ''
        for v, count in vars:
            if count == 1:
                temp += '*' + v
            else:
                temp += ('*' + v + '^' + str(count))
        # print(temp)
        if round(c, 3) > 0:
            e += str(c) + temp + ' + '
    return target_feature + '=' + e + '%.3f' % (round(intercept, 3))


def rebuild_equations(equations):
    """ rebuild the equations after modifying them to satisfy the requirement in a SDM model,
    then forward them into it"""
    equs = {}

    for process_var, info in equations.items():
        e = ''
        for coef, exp_var in info[:-1]:
            if e == '':
                e += "%s * %s" % (coef, exp_var)
            else:

                if coef > 0:
                    e += '+' + " %s * %s " % (coef, exp_var)
                else:
                    e += " %s * %s " % (coef, exp_var)
        # e = " + ".join("%s * %s" % (coef, exp_var) for (coef, exp_var) in info[:-1])
        intercept = info[-1]
        if intercept != 0:
            equs[process_var] = (e + '%+.3f' % (round(intercept, 3)))
        else:
            equs[process_var] = e
    return equs
