import pysd

def update_sdm(sd, equations, sf_vars, data_var, path_to_model):
    """

    :param sd: given SD log
    :param equations: the equation information that need to forward into a sdm model
    :param sf_vars: stock and flow variables
    :param path_to_model: the absolute path of given sdm model
    :return: the updated sdm model
    """
    model = pysd.read_vensim(path_to_model)

    # name mapping of variables in equations and variables in model
    name_relation = dict(zip(model.doc()['Real Name'], model.doc()['Py Name']))
    stocks = [name_relation[i[0]] for i in sf_vars]

    ans = {}
    for key, val in equations.items():
        for k in name_relation.keys():
            if k in val:
                val = val.replace(k, '{}()'.format(name_relation[k]))
        if name_relation[key] not in stocks:
            ans[name_relation[key]] = val

    length = len(sd)

    path_to_save = list(path_to_model)
    while path_to_save:
        cur = path_to_save.pop()
        if cur == '/':
            break
    path_to_save = ''.join(path_to_save)
    path_to_save += '/updated_model.py'

    with open(path_to_save, 'w') as m:
        with open(path_to_model[:-3] + 'py', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                for var, equ in ans.items():
                    # print('var',var)
                    # if var not in stocks:
                    if 'def {}():'.format(var) in line:
                        lines[i + 3] = "Original Eqn: b'{}'\n".format(equ)
                        lines[i + 10] = '    return {}'.format(equ)
                        lines[i + 11] = ''

                if 'def final_time():' in line:
                    lines[i + 10] = '    return {}'.format(length)

                if 'def initial_time():' in line:
                    lines[i + 10] = '    return 1'

                for var in data_var:
                    var_in_py = name_relation[var]
                    if 'def {}():'.format(var_in_py) in line:
                        lines[i +
                              3] = "Original Eqn: b'{}'\n".format(sd[var][1])
                        lines[i + 10] = '    return {}'.format(sd[var][1])
                        lines[i + 11] = ''

                m.writelines(line)
    updated_model = pysd.load(path_to_save)
    return updated_model


def run_sdm(model, data_var, sd):
    """

    :param model: the sdm model that need to run
    :param data_var: the data variables indicates which variables should read directly from local
    :param sd: the given SD-Log
    :return: the simulated results
    """
    model = pysd.read_vensim(model)
    model.run()

    data_d = {}
    for data_v in data_var:
        data_d[data_v] = sd[data_v]

    model.set_components(params=data_d)
    stocks = model.run()
    return stocks
