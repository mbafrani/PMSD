import pandas as pd
import random
import pysd
# random.seed(10)


class MoDeL:

    def __init__(self):
        """ initialize an MoDeL object"""
        self.equations = []
        # self.controls = []
        self.sketches = {}

    def equations_and_links(
            self,
            sd_log,
            math_equations,
            sf_vars,
            data_vars):
        """
        write equation information for every variable entity in the System Dynamics Model

        ---- auxiliary  variable ----
         Characteristic Time=
         10+0.5*Outside Temperature
         ~	Minutes
         ~		|

         ---- stock variable ----
         Teacup Temperature= INTEG (
         -Heat Loss to Room,200)
         ~
         ~		|

         ---- data variable ----
         Arrival rate:INTERPOLATE::=
         GET XLS DATA( 'active2012_2H.xlsx','Sheet2','A','B2' )
         ~
         ~		|

        :param equs_info: mathematical formula for process variables in the SD model, e.g.,
        {'Finish rate2H': '0.9530629623326526 * Arrival rate2H+ 1.1468984268415774 * Num of unique resource2H
        -0.005541249149232541 * Service time per case2H  -0.310221629036053 * Num in process case2H -6.964'}
        :param sf_vars: stock variable and its flow variables. dictionary {stock:[[inflow], [outflow]]}
        :param data_vars: data variables whose value are read locally, a set object.
        :return:
        """

        stocks = list(sf_vars.keys())
        # flows = [f for _,flow in sf_vars for f in flow]

        # write equation information for auxiliary variable
        for process_variable, equ in math_equations.items():
            # stock variable has a fix mathematical form
            if process_variable not in stocks and process_variable not in data_vars:
                line = process_variable + '=' + equ
                self.equations.append(line)
                self.equations.extend(['~', '~', '|', ''])

        # write equation information for stocks
        for stock in sf_vars.keys():
            stock_equation = list(math_equations[stock])
            initial_stock_val = []
            cur = ''
            while (stock_equation and cur != '+'):
                ch = stock_equation.pop()
                if ch != '+':
                    initial_stock_val.append(ch)
                cur = ch
            initial_stock_val = ''.join(initial_stock_val[::-1])
            stock_formula = ''.join(stock_equation)

            line = stock + \
                '= INTEG({}, {})'.format(stock_formula, initial_stock_val)
            self.equations.append(line)
            self.equations.extend(['~', '~', '|', ''])

        # write equation information for data variable
        for data_variable in data_vars:
            # line = datavar + ':INTERPOLATE::=GET XLS DATA()'
            line = data_variable + '= 0'
            self.equations.append(line)
            self.equations.extend(['~', '~', '|', ''])

    def sketch_informaiton(self, equs_info, sf_vars, data_vars):
        """
        add sketch information into the .mdl file, mainly we care about the variable entities and links in the model
        :param equs_info: equations information that should be saved into the variable entity to run the model
        :param sf_vars: stock and flow variables, dictionary {stocks:[[inflow], [outflow]]]
        """
        id = 1

        # (x,y) is the word position, w, h is the width and height of the word
        x, y, w, h = 700, 500, 100, 50
        step_size = 200

        stock_tail = ',3,3,0,0,0,0,0,0'
        var_tail = ',8,3,0,0,0,0,0,0'
        flow_tail = ',40,3,0,0,-1,0,0,0'
        valves_tail = ',34,3,0,0,1,0,0,0'
        comment_tail = ',0,3,0,0,-1,0,0,0'
        arrow_tail = ',0,0,22,0,0,0,-1--1--1,,1|'

        temp = data_vars.copy()
        # stocks = list(sf_vars.keys())

        for stock, (inflows, outflows) in sf_vars.items():
            self.sketches['10,{},{}'.format(id, stock)] = '{},{},{},{}'.format(
                x, y, w, h) + stock_tail
            cur_stock_id = id
            id += 1
            if inflows:
                for inflow_variable in inflows:
                    # if inflow and inflow not in ['n','N']:
                    self.sketches['12,{},48'.format(id)] = '{},{},{},{}'.format(
                        x - 2 * step_size, y, w, h) + comment_tail  # comment
                    cur_comment_id = id
                    id += 1
                    self.sketches['1,{}'.format(id)] = '{},{},4'.format(
                        id + 2, cur_stock_id) + arrow_tail + '({},{})|'.format(x - step_size, y)
                    id += 1
                    self.sketches['1,{}'.format(id)] = '{},{},100'.format(
                        id + 1, cur_comment_id) + arrow_tail + '({},{})|'.format(x - step_size - 50, y)
                    id += 1
                    self.sketches['11,{},48'.format(id)] = '{},{},6,8'.format(
                        x - step_size, y) + valves_tail
                    id += 1
                    self.sketches['10,{},{}'.format(id, inflow_variable)] = '{},{},{},{}'.format(
                        x - step_size, y + 20, int(w * 0.8), int(h * 0.8)) + flow_tail
                    id += 1
                    if inflow_variable in temp:
                        temp.remove(inflow_variable)
            if outflows:
                for outflow_variable in outflows:
                    # if outflow and outflow not in ['n','N']:
                    self.sketches['12,{},48'.format(id)] = '{},{},{},{}'.format(
                        x + 2 * step_size, y, w, h) + comment_tail  # comment
                    cur_comment_id = id
                    id += 1
                    self.sketches['1,{}'.format(id)] = '{},{},100'.format(
                        id + 2, cur_stock_id) + arrow_tail + '({},{})|'.format(x + step_size, y)
                    id += 1
                    self.sketches['1,{}'.format(id)] = '{},{},4'.format(
                        id + 1, cur_comment_id) + arrow_tail + '({},{})|'.format(x + step_size - 80, y)
                    id += 1
                    self.sketches['11,{},48'.format(id)] = '{},{},6,8'.format(
                        x + int(0.95 * step_size), y) + valves_tail
                    id += 1
                    self.sketches['10,{},{}'.format(id, outflow_variable)] = '{},{},{},{}'.format(
                        x + int(0.95 * step_size), y + 20, int(w * 0.8), int(h * 0.8)) + flow_tail
                    id += 1
                    if outflow_variable in temp:
                        temp.remove(outflow_variable)

            # update stock pivot position if we have more than one stock
            # variable
            x += step_size
            y += step_size
        # print('-'*100)
        # print(self.sketches)
        # print('done!')
        a, b, c, d = 1000, 1000, 100, 50
        f = random.uniform(0.5, 1.5)
        g = random.uniform(0.5, 0.8)
        p = random.uniform(1, 2)

        # stocks = [stock for stock, _ in sf_vars]
        # flows = [f for _, flow in sf_vars for f in flow]

        stock_variable, flow_variables = list(), list()
        for stock, [inflow, outflow] in sf_vars.items():
            stock_variable.append(stock)
            if inflow and outflow:
                for cur_flow in inflow + outflow:
                    flow_variables.append(cur_flow)
            elif inflow:
                for cur_flow in inflow:
                    flow_variables.append(cur_flow)
            else:
                for cur_flow in outflow:
                    flow_variables.append(cur_flow)
        #
        for var in equs_info.keys():  # add auxiliary variables in the model
            if var not in stock_variable and var not in flow_variables:
                self.sketches['10,{},{}'.format(id, var)] = '{},{},{},{}'.format(
                    int(a * f), int(b * f), int(c * g), int(d * g)) + var_tail
                id += 1

        if temp:  # add data variables in the model
            for var in temp:
                if var not in ['n', 'N']:
                    self.sketches['10,{},{}'.format(id, var)] = '{},{},{},{}'.format(
                        int(a * f * p), int(b * f * p), int(c * g * p), int(d * g * p)) + var_tail
                    id += 1

    def write2file(self, sd, path_to_file):
        """
        write all content into a mdl file

        :param sd: the sd-log
        :param path_to_file: absolute path to the output sdm model with an extension .mdl
        """
        mdl = open(path_to_file, 'w')

        # write equation information
        mdl.writelines('{UTF-8}' + '\n')
        for line in self.equations:
            mdl.writelines(line + '\n')

        # write control information
        control_info = open('contro_info.txt', 'r')
        new_line = 'FINAL TIME  = {}'.format(len(sd))

        for line in control_info.readlines():
            if 'FINAL TIME  = ' in line:
                mdl.writelines(new_line + '\n')
            else:
                mdl.writelines(line)

        mdl.writelines('\n')

        # write sketch header
        sketch_header = open('sketch_header.txt', 'r')

        for line in sketch_header.readlines():
            mdl.writelines(line)
        mdl.writelines('\n')
        for key, val in self.sketches.items():
            line = key + ',' + val
            mdl.writelines(line + '\n')


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

    # print(data_d)
    model.set_components(params=data_d)
    stocks = model.run()
    return stocks
