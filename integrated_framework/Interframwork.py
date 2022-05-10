import os
import time
from shutil import copyfile

import pandas as pd
import warnings
import matplotlib.pyplot as plt
from pyvis.network import Network
from sklearn.model_selection import train_test_split

from .training_models import *
from .result_visualization import *
from .data_preprocessing import *
from .equation_processing import *
from .user_interactions import GetInteraction
from .system_dynamics import SDM
from .equation2df import Equs2df
from .equation2cld.GenCLD import CausalLoopDiagram
from .equation2mdl.CreateMDL import MoDeL
from .dataframes_corrs import CheckCorrelation
from .dataframes_corrs import CreateDataframe
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
warnings.filterwarnings('ignore')

cwd = os.getcwd()
class InteFramework:

    def __init__(self):
        """
        initialize the interFramework object
        """
        # self.path = pathToFile
        # self.var = processVar
        self.best_equation = {}  # contain all best equations
        self.models = {
            'Linear Regression': BestSubsetSelection.best_subset_selection,
            'Lasso Regression': RegularizationBasedRegression.lasso_regression,
            'Ridge Regression': RegularizationBasedRegression.ridge_regression,
            'Elastic Net': RegularizationBasedRegression.elastic_net,
            'Support Vector Regression': SVRegression.call_svr}
        self.updated_equation = None
        self.namespace = {}
        self.features = {}
        self.var_fail_to_get_equations = set()
        self.mean_values = {}

    def read_sd(self, path_to_file):
        """
        given file path, get data

        ------
        :return: the SD log in a data frame
        """
        # path_to_file = self.path
        df = pd.read_csv(path_to_file)
        df.index += 1    # row index start from 1, easy to do compare final model simulated result and data in df

        self.features = list(df.columns)
        for feature in self.features:
            self.namespace[feature] = feature.strip().replace(
                ' ', '_').lower()  # replace space in feature names
        # rename the input SD-Log
        df.rename(columns=self.namespace, inplace=True)
        self.mean_values = {k: df[k].mean() for k in self.namespace.values()}

        return df

    def visualize_data(self, df):
        """
        visualize the input SD-Log
        :return: None
        """
        fig, axes = plt.subplots(len(df.columns), 1, figsize=(20, 15), dpi=120)
        for i, ax in enumerate(axes.flatten()):
            vals = df[df.columns[i]]
            ax.plot(vals, color='blue', linewidth=1)
            # Decorations
            ax.set_title(df.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)
        plt.tight_layout()
        plt.show()
        return

    def get_data(self, df, target_variable=None):
        """
        :param df: SD-Log in data frame
        :param target_variable: the process variable
        :return: predictors values (x) and process variable values (y)
        """
        x, y = df.drop(target_variable, axis=1), df[target_variable]
        return x, y

    def drop_variables(self, df, variables=None):
        """

        :param df: SD-Log in data frame
        :param variables: the process variables that we want to delete, in string form or list of strings
        :return: SD-Log without unwanted variables
        """
        return df.drop(variables, axis=1, inplace=True)

    def data_spliting(self, xdata, ydata, related_predictors=None):
        """

        :param xdata: predictors values in SD-Log
        :param ydata: process variable values in SD-Log
        :param related_predictors: related variables which have a strong correlation again the process variable
        :return: data in training and testing set
        """
        if related_predictors is None:
            x = xdata
        else:
            x = xdata[related_predictors]
        x_train, x_test, y_train, y_test = train_test_split(
            x, ydata, random_state=374639)
        return x_train, x_test, y_train, y_test

    def get_covnerted_dfs(self):
        return self.dataframes

    def get_updated_equations(self):
        return self.updated_equation

    def find_relations(self, sd_log, target_feature, threshold, path_to_file):
        """
        find the best relation for given sd-log and process variable if exists

        :param sd_log: input SD-log
        :param target_feature: the selected process variable
        :param threshold:  a pre-defined threshold to select related explanatory variables by using a DCC matrix
        :param path_to_file: path to the user PC's desktop where to save the running result

        """

        # get predictors and target
        # x: dataframe, y: pd series
        x, y = self.get_data(sd_log, target_feature)

        # select related predictors based on DCC matrix
        dis_cor = CalculatDccMatrix.calculate_dcc_matrix(
            x, y)

        # select process variables which have a strong distance correlation
        # coefficient
        independent_vars = CalculatDccMatrix.select_predictors(
            dis_cor, threshold)

        if not independent_vars:  # no related IVs against the target variable, i.e, target_var = 0* explanatory variables
            self.best_equation[target_feature] = [
                [], sd_log[target_feature][1], [], '']
            msg = 'The best potential equation of feature {} is trivial, no related explanatory variables are detected'.\
                format(target_feature)
            equations = '{} = {}'.format(
                target_feature, sd_log[target_feature][1])
            print(msg)
            print(equations)
            return msg, equations

        # get train and test dataset
        # x_data: dataframe, y: pd.series
        x_data = x[independent_vars]
        x_train, x_test, y_train, y_test = DataPreprocessing.data_spliting(
            x_data, y)

        # since we try all predictors' combi with best subset selection
        # algorithm, here we use the whole input SD-Log especially for best_subset_selection_based_regression
        # i.e., try all variable combinations
        xtrain, ytrain, xtest, ytest = DataPreprocessing.data_spliting(
            x, y)

        # model train
        training_res = dict()

        # training...
        rmse_list = {}
        # curve_fitting_rmse = float('inf')
        for model, fct in self.models.items():
            print('Start training a {} model now...'.format(model))
            if model == 'linear regression':
                # less than 15 feature, we use the best subset to find the model, because it won't spend too much
                # time to do it
                if sd_log.shape[1] <= 15:
                    model_summary = fct(xtrain, xtest, ytrain, ytest)
                    print('Training is done.')
                    training_res[model] = model_summary
                else:
                    model_summary = AutoSelectionBasedRegression.auto_selection_based_regression(
                        xtrain, xtest, ytrain, ytest)  # otherwise, we use auto selection-based regression to speed up
                    print('Training is done')
                    training_res['stepwise linear regression'] = model_summary
            else:
                try:
                    model_summary = fct(x_train, x_test, y_train, y_test)
                    print('Training is done.')
                    training_res[model] = model_summary
                except:
                    pass

        for model, val in training_res.items():
            if model == 'Support Vector Regression':
                if isinstance(val[0], str):
                    rmse_list['{} with {} kernel'.format(
                        model, val[0])] = val[-1]
                else:
                    rmse_list['Support Vector Regression with linear kernel'] = val[-1]
            else:
                rmse_list[model] = val[3][1]  # {model:the rmse_train_value}

        # now we select the best model from the trained model set
        ans = GetBestModel.select_best_model(
            training_res, self.mean_values, target_feature)
        if not ans:
            self.var_fail_to_get_equations.add(target_feature)
            msg = 'Failed to find equation for {} due to overfitting or high training errors, recommend as a data' \
                  'variable in the final System dynamics model.'.format(target_feature)
            equ = 'null'
            return msg, equ

        else:
            coef, intercept, IVs, min_rmse, selected_training_method = ans[
                0], ans[1], ans[2], ans[3], ans[-1]

            PlotRMSE.plot_rmse(
                rmse_list, target_feature, path_to_file)

        # if the min_rmse value (rmse_value on test set) is grater than the mean value of target feature,
        # then the equation maybe too complex to use just one model to express,
        # call curve fitting now

            if min_rmse > 2 * \
                    abs(sd_log[target_feature].mean()):    # in case the mean value is negative,use abs()
                print(
                    'The model error seems too high, please fit a curve with some good guesses')
                time.sleep(2)
                print('Start fitting curves...')
                curve_equation, curve_fitting_rmse = FittingCurves.final_curve_fitting(
                    x_train, x_test, y_train, y_test)
                rmse_list['curve fitting'] = curve_fitting_rmse
                print('Fitting is done.')

                if curve_fitting_rmse < min_rmse:
                    # if the rmse value of fitted line is better than the last rmse
                    # value, then we take it
                    msg = 'The best potential equation of feature {} comes from Curve Fitting'.format(
                        target_feature)

                    self.best_equation[target_feature] = [
                        curve_equation, 'Curve fitting']  # 这里可能有问题
                    return msg, curve_equation
                else:
                    time.sleep(2)
                    print(
                        'Failed to find an equation for process variable {} because the RMSE error of fitted curve is '
                        'still to high'.format(target_feature))
                    self.var_fail_to_get_equations.add(target_feature)
                    msg = 'Failed to find equation for {} due to overfitting or high training errors, recommend as a ' \
                          'data variable in the final System dynamics model.'.format(target_feature)
                    equ = 'null'
                    return msg, equ

            else:

                # if the results from polynomial regression, then the variables may contains [var^2, var1 var2 ... var_n]
                # (indicates that var1 * var2 * ... * var_n), we need to handle this case
                if selected_training_method == 'polynomial regression':
                    reranged_var_list = []
                    for var in IVs:
                        temp = []
                        cur = var.strip().split(' ')
                        if len(cur) > 1:   # like [var1 var2 var3^2]
                            for idx, e in enumerate(cur):
                                if e[-2] != '^':  # var1 type: string
                                    temp.append((e, 1))
                                else:
                                    # var1^2 --> var1 * var1
                                    temp.append((e[:-2], int(e[-1])))
                        else:                    # like [var1], [var2^2]
                            if cur[0][-2] != '^':   # like [var1]
                                temp.append((cur[0], 1))
                            else:                      # [var2^2]
                                temp.append((cur[0][:-2], int(cur[0][-1])))
                        reranged_var_list.append(temp)

                    self.best_equation[target_feature] = [
                        coef, intercept, reranged_var_list, 'polynomial regression']
                    msg = 'The best potential equation of feature {} comes from polynomial regression'.format(
                        target_feature)
                    return msg, GetBestModel.print_polynomail_equation(
                        target_feature, coef, reranged_var_list, intercept)
                else:
                    self.best_equation[target_feature] = [
                        coef, intercept, IVs, selected_training_method]
                    msg = 'The best potential equation of feature {} comes from {}'.format(
                        target_feature, selected_training_method)
                    return msg, GetBestModel.pretty_print_linear(
                        target_feature, coef, intercept, IVs)

    def call_framework(self, sd_log, feature=None, threshold=0.05):

        warnings.filterwarnings("ignore")

        # get the first sight of input data
        # self.visualize_data(sd_log)

        # input("Press Enter to continue...")
        # dropped_features = GetInteraction.drop_column_switcher(sd_log)
        # if dropped_features:
        #     self.drop_variables(sd_log, dropped_features)

        # self.visualize_data(data)
        sd_log = DataPreprocessing.remove_trivial_parameters(sd_log)
        sd_log = DataPreprocessing.remove_missing_values(sd_log)
        sd_log = DataPreprocessing.remove_outliers(sd_log)

        # path for saving equation and info
        #desktop_path = os.path.join(os.path.expanduser('~'), "Desktop")
        #directory = 'running_results'
        #path_to_file = os.path.join(desktop_path, directory)
        #os.makedirs(path_to_file, exist_ok=True)
        path_to_file = os.path.join( str(cwd), 'Outputs')
        file_to_open = os.path.join( str(cwd), 'Outputs', 'equations.txt')

        # text file to store equations
        f = open(file_to_open, 'w')
        # poly_label = GetInteraction.poly_regression_switcher()

        poly_label = 'y'  # reduce user interaction, set the poly_label as 'y', i.e., we use the polynomial regression
        if feature is not None:

            if poly_label in ['y', 'Y']:
                self.models['polynomial regression'] = PolyRegression.train_polynomial_model
            feature = list(feature.strip().split(','))
            for fea in feature:
                # rename variables, if user given features like 'Arrival
                # rate1W'
                target_fea = fea.strip().replace(' ', '_').lower()
                msg, equation = self.find_relations(
                    sd_log, target_fea, threshold, path_to_file)
                f.write('-' * 50 + '\n')
                f.write(msg + '\n')
                f.write(equation + '\n')
                f.write('-' * 50 + '\n')
        else:
            feature_list = list(sd_log.columns)

            if poly_label in ['y', 'Y']:
                self.models['polynomial regression'] = PolyRegression.train_polynomial_model
            for fea in feature_list:
                print('-' * 50)
                print('Regarding feature {}'.format(fea))
                print('-' * 50)

                m, e = self.find_relations(
                    sd_log, fea, threshold, path_to_file)
                f.write('-' * 50 + '\n')
                f.write(m + '\n')
                f.write(e + '\n')
                f.write('-' * 50 + '\n')

        # save cleaned data as csv file in local for further simulation use
        # re-name again to get it back
        sd_log.to_csv(
            os.path.join(
                path_to_file,
                'cleaned_data.csv'),
            index=False)
        print('all done!')

    def write_cfd(self,corr_df):
        cwd = os.getcwd()
        map_dict={}
        corr_df_dict = corr_df.to_dict('dict')
        tempvarnames = ['a','b','c','d','e','f','g','h','i','j']
        copyfile(os.path.join('integrated_framework','testDFD.mdl'),os.path.join('integrated_framework',"newtestDFD.mdl"))
        f = open(os.path.join( 'integrated_framework',"newtestDFD.mdl"), 'r')
        filedata = f.read()
        f.close()
        i = 0
        for kv in corr_df.index.values.tolist():
            if "*" not in kv or "^" not in kv:
                try:
                    filedata = filedata.replace(','+str(tempvarnames[i]+','), ','+str(kv)+',')
                except:
                    pass
            f = open(os.path.join('integrated_framework',"newtestDFD.mdl"), 'w')
            f.write(filedata)
            f.close()
            i +=1

        with open(os.path.join('integrated_framework',"newtestDFD.mdl"), 'r+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                if "*" not in km or "^" not in km:
                    tempstr = ''
                    for kmm, vmm in vm.items():
                        if vmm != 0 and tempstr == '':
                            tempstr = tempstr+str(kmm)
                        elif vmm != 0 and tempstr != '':
                            tempstr = tempstr + ',' + str(kmm)

                    cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')


            cfdfile.close()

        with open(os.path.join('integrated_framework',"newtestDFD.mdl"), 'r+') as f:
            content = f.read()
            try:
                tempplace= content.index(':L<%^E!@')
                f.seek(tempplace)
                f.write("")
            except:
                pass

        print ('done')

        #TODO: visualize CFD python

        corr_df[corr_df == 0] = 0
        G = Network(height="800px",
                 width="800px",directed=True)
        corr_dft = corr_df.T

        for act in corr_df.columns:
            if "*" not in act or "^" not in act:
                G.add_node(act, shape='box', label=str(act),borderWidth=0,color = 'white')
                temp_in = corr_df[act]
                for intemp in temp_in.iteritems():
                    G.add_node(intemp[0], shape='box', color='white',label=str(intemp[0]))
                    if corr_df[act][intemp[0]] ==0:
                        print('no edge')
                    elif corr_df[act][intemp[0]] > 0:
                        edgelabel= '+'
                        G.add_edge(intemp[0], act, color='blue', label=edgelabel,
                                   title=str(corr_df[act][intemp[0]]))
                    elif corr_df[act][intemp[0]] < 0:
                        edgelabel = '-'
                        G.add_edge(intemp[0],act,color= 'blue',label=edgelabel,title=str(corr_df[act][intemp[0]]))
        G.toggle_physics(False)
        G.save_graph(os.path.join(cwd,'templates','mygraph1.html'))
        #G.save_graph(str(os.getcwd())+"\\templates\mygraph.html")
        #G.save_graph(r"C:\Users\mahsa\PycharmProjects\SharedProjectForReviewCoopis2019\templates\cfdgraph.html")

        #TODO write the relation function into text for further use
        with open(os.path.join('integrated_framework',"relationinCFD.txt"), 'w+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                if "*" not in km or "^" not in km:
                    tempstr = ''
                    for kmm, vmm in vm.items():
                        if vmm != 0 and tempstr == '':
                            tempstr = tempstr+str(kmm)
                        elif vmm != 0 and tempstr != '':
                            tempstr = tempstr + ',' + str(kmm)

                    km=km.replace(" ",'')
                    tempstr=tempstr.replace(" ", '')
                    cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')

        with open(os.path.join('ModelsFormat',"relationinCFD.txt"), 'w+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                if "*" not in km or "^" not in km:
                    tempstr = ''
                    for kmm, vmm in vm.items():
                        if vmm != 0 and tempstr == '':
                            tempstr = tempstr+str(kmm)
                        elif vmm != 0 and tempstr != '':
                            tempstr = tempstr + ',' + str(kmm)
                    km = km.replace(" ", '')
                    tempstr = tempstr.replace(" ", '')
                    cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')

        return

def run(path, model_path=None, cyclefree=int):
    # initialize a integrated framework object and find the best equations for
    # variables in given SD-Log
    obj = InteFramework()
    data = obj.read_sd(
        path_to_file=path)
    dcc_df, lag_corr_df = CheckCorrelation.check_correlation(data)
    # pick the strong correlation from dcc and lag
    output_df = CheckCorrelation.pick_strong_correlations(dcc_df, lag_corr_df)
    output_df1=CheckCorrelation.pick_strong_corr(dcc_df,lag_corr_df)
    final_corr_out=CheckCorrelation.check_real_corr(output_df,output_df1,data)
    #obj.write_cfd(final_corr_out.T)
    # print(data.columns)
    obj.call_framework(data)

    # all best equations that we found by the framework
    all_best_equations = obj.best_equation

    # specify stock variables and flow variables

    # # specify component
    # sf_vars = GetInteraction.get_stock_and_flow_variables(
    #     data)  # specify the stock and flow variables
    # # specify the data variables if you know it already, otherwise the
    # # framework will try to detect it
    # data_var = GetInteraction.get_data_variables(data)

    # sf_vars, data_var = None, None  # from now on, we don't care about
    # stocks, flows and so on...

    # remove very tiny coefficient, this is specially for polynomial regression. Because some coefficient of polynomial
    # regression is extremely small, e.g. 1.2345e-13
    all_best_equations_without_tiny_coef = ProcessEquations.pre_processing_equations(
        obj.mean_values, all_best_equations)
    #
    # print('remove tiny coefficient')
    # print(all_best_equations_without_tiny_coef)

    # create data frames for discovered equations
    from .dataframes_corrs import CreateDataframe
    raw_df, raw_causal_effect_df, all_lines = CreateDataframe.create_dataframes_from_equations(
        all_best_equations_without_tiny_coef, obj.namespace)
    # simplify the created data frame, just keep essential information
    simplified_df, simplified_causal_df = CreateDataframe.simplify_df(raw_df,cyclefree)
    obj.write_cfd(simplified_df.T)
    # check the distance correlation and lag correlation
    dcc_df, lag_corr_df = CheckCorrelation.check_correlation(data)
    # pick the strong correlation from dcc and lag
    output_df = CheckCorrelation.pick_strong_correlations(dcc_df, lag_corr_df)
    #obj.write_cfd(output_df)
    # process all best equation in further, this time eliminate coefficient smaller than pre-defined threshold
    # and also get line information to build CLD
    # processed_equations, stock_lines, lines_with_bigger_coefficient, lines_with_smaller_coefficient = ProcessEquations.process_equations_new(
    #     data, all_best_equations_without_tiny_coef, sf_vars, data_var)
    #
    # removed_edges = ProcessEquations.edges_2be_removed(
    #     lines_with_bigger_coefficient)  # edges need to be removed
    #
    # # create dataframes
    # # stock lines + lines_with_bigger_coefficient = CLD before processing
    # # stock lines + lines_with_bigger_coefficient -
    # # (lines_with_smaller_coefficient + removed_lines) = CLD after processing
    # df_before_processing, df_after_processing = Equs2df.lines_to_df(
    #     obj.namespace, stock_lines, lines_with_bigger_coefficient, lines_with_smaller_coefficient, removed_edges)
    #
    # desktop_path = os.path.join(os.path.expanduser('~'), "Desktop")
    # directory = 'running_results'
    # path_to_file = os.path.join(desktop_path, directory)
    # os.makedirs(path_to_file, exist_ok=True)
    #
    # df_after_processing.to_csv(
    #     os.path.join(
    #         path_to_file,
    #         'CLD_information_before_processing.csv'),
    #     index=False)
    #
    # df_after_processing.to_csv(
    #     os.path.join(
    #         path_to_file,
    #         'CLD_information_after_processing.csv'),
    #     index=False)
    #
    # # remove edges to get a cycle-free (but a stock-in-side cycle is ok) graph
    # # build by processed_equations
    # updated_processed_equations = ProcessEquations.update_processed_equations(
    #     obj.mean_values,
    #     processed_equations,
    #     removed_edges)
    #
    # # detect data variables if needed, remember to remove trivial process variables at the very beginning otherwise these
    # # variables will be treated as data variables (data variables is variable
    # # has no inbound link in a CLD)
    # if not data_var:
    #     data_var = ProcessEquations.detect_data_variables(
    #         obj.namespace, updated_processed_equations, obj.var_fail_to_get_equations)
    #
    # print('The data variables in the created SDM is', data_var)
    # # get mathematical equations to run a SDM
    # equations_in_mathematical = ProcessEquations.rebuild_equations(
    #     updated_processed_equations, data_var)
    #
    # generate CLD
    cld = CausalLoopDiagram()
    # cld diagram generated use equations before pre-processing
    cld.get_links_by_sign(raw_causal_effect_df)
    cld.add_causal_links()
    cld.draw(os.path.join(cwd,'static','images','cld_before.png'))

    cld2 = CausalLoopDiagram()  # cld diagram generated use equations after pre-processing
    cld2.get_links_by_sign(simplified_causal_df.T)
    cld2.add_causal_links()
    cld2.draw(os.path.join(cwd,'static','images','cld_after.png'))
    #
    # # create a System Dynamics model file with extension .mdl so that user can
    # # use it in software 'Vensim'
    # mdl = MoDeL()
    # mdl.equations_and_links(data, equations_in_mathematical, sf_vars, data_var)
    # mdl.sketch_informaiton(equations_in_mathematical, sf_vars, data_var)
    # mdl.write2file(data, 'model.mdl')
    #
    # # RUN model to compare the simulated result and data in given SD-Log
    # stocks = SDM.run_sdm('model.mdl', data_var, data)
    #
    # # dummy stocks
    # if model_path:
    #     dummy_model = pysd.read_vensim(model_path)
    #     dummy_stocks = dummy_model.run()
    #     dummy_stocks.rename(columns=obj.namespace, inplace=True)
    #     print(dummy_stocks)
    #     PlotSimulatedresult.plot_simulated_res(stocks, data, dummy_stocks)
    #
    # PlotSimulatedresult.plot_simulated_res(stocks, data)


#if __name__ == '__main__':
    #run(path=r'C:\Users\pads\Desktop\PycharmProjects\BinEQfinderSourceCode-thesis-master\thesis-master\data\BPI2017\BPI2017General_7D_sdlog.csv')
    #run(path=r'C:\Users\pads\Desktop\PycharmProjects\BinEQfinderSourceCode-thesis-master\thesis-master\data\BPI2017\BPI2017Active_1D_sdlog.csv')
    #run(path=r'C:\Users\pads\Desktop\Work&Teaching\WeeklyReports\ModelSuppotingPaperBIS2020EXTENSIONEQ\Extention\testSDvensimBPR4.csv')
    #run(path=r'C:\Users\pads\Desktop\Work&Teaching\WeeklyReports\ModelSuppotingPaperBIS2020EXTENSIONEQ\Extention\TwoOrganizationHandover.csv')