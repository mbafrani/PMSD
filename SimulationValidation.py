
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysd
from collections import defaultdict
import seaborn as sns
from scipy.stats import variation, ks_2samp, shapiro

class SimVal:

    def read_model(self, sd_model,sd_log_file):
        model = pysd.read_vensim(sd_model)
        x = dir(model.components)
        stock_var = []

        for i in x:
            if i[:5] == 'integ':
                stock_var.append(i[6:])

        variable_involeved_val = model.run()
        variable_involeved = variable_involeved_val.columns.str.replace(' ', '_')
        df_real_val = pd.read_csv(sd_log_file)

        var_dict_params = {}
        for name in variable_involeved:
            if name in df_real_val.columns and name.lower() not in stock_var:
                v = df_real_val[name]
                # v = v[:12]
                vname = name.lower()

                var_dict_params.update({vname: v})

        stocks = model.run(params=var_dict_params)
        sim_values = stocks

        sim_values = sim_values.where(sim_values >0, other=0)
        variables_names = stocks.columns
        return sim_values

    def creat_real_sim_dict(self, sd_log, sim_values):
        real_sim_dict = defaultdict(list)
        real_values_sd_log=pd.read_csv(sd_log)

        sim_values.columns = sim_values.columns.str.replace(' ', '_')
        real_values_sd_log.columns = real_values_sd_log.columns.str.replace(' ', '')
        real_val_names = real_values_sd_log.columns

        for rname in real_val_names:
            rname.replace("_", "")
            rname.replace("_", "")


            if rname in sim_values.columns:
                sim_size = len(sim_values[rname])
                real_siz = len(real_values_sd_log[rname])
                if sim_size <= real_siz:
                    real_list = real_values_sd_log[rname]
                    real_sim_dict[rname].append(real_list[:sim_size])
                    real_sim_dict[rname].append(sim_values[rname])
                else:
                    sim_list = real_values_sd_log[rname]
                    real_sim_dict[rname].append(real_values_sd_log[rname])
                    real_sim_dict[rname].append(sim_list[:real_siz])

        return real_sim_dict

    def validate_results(self,real_sim_dict):

            plt.rcParams.update({'font.size': 10})
            val_image_names =[]
            plt.tight_layout()
            for vname, vvalue in real_sim_dict.items():
                plt.figure()
                simulation_list = []
                simulation_list = vvalue[1].values
                real_list = []
                real_list = vvalue[0]

                if len(simulation_list) != len(real_list):
                    real_list = real_list[:(len(simulation_list)-len(real_list))].values
                # data description
                plt.subplot(3, 3, 1)
                plt.tight_layout()
                plt.gca().set_title("reality vs simulation",size= 8)
                data = [real_list, simulation_list]
                #plt.plot(np.cumsum(real_list))
                plt.plot(real_list,)
                plt.plot(simulation_list,'--')

                plt.subplot(3, 3, 4)
                plt.tight_layout()
                plt.gca().set_title("distribution of reality vs simulation",size = 8)
                sns.distplot(real_list,color="blue", hist=False)
                sns.distplot(simulation_list,color="orange", kde_kws={'linestyle':'--'})
                statistical_compare = pd.DataFrame(
                    [[np.mean(real_list), np.std(real_list), variation(real_list)],
                     [np.mean(simulation_list), np.std(simulation_list),variation(simulation_list)]],
                      index=['reality', 'simulated'],columns=['mean', 'STD', 'CV'])

                plt.subplot(3, 3,(2,3) )
                plt.tight_layout()
                plt.axis('off')
                plt.text(0.1, 0.2, "reality:"+
                         str(statistical_compare.loc['reality',:])+"\n simulation:"+
                         str(statistical_compare.loc['simulated', :]), fontsize=9)

                # Test Distribution of Data
                sta, p_value = ks_2samp(real_list, simulation_list)
                plt.subplot(3, 3, (5,6))
                plt.tight_layout()

                plt.axis('off')
                if p_value >= 0.05:
                    plt.text(0.1, 0.6, "P value of the Kolmogrov Test:\n" + str(round(p_value,2))+ " (Two distributions are similar)", fontsize=9)
                else :
                    plt.text(0.1, 0.6,"P value of the Kolmogrov Test:\n" + str(round(p_value,2)) + " (Two distributions are not similar)",fontsize=9)

                # Test PairWise
                diff_list = []
                diff_list = np.array(diff_list)
                diff_list = abs(np.array(real_list) - np.array(simulation_list))
                plt.subplot(3, 3,7)
                plt.tight_layout()
                plt.gca().set_title("Distribution of Difference",size = 9)
                if np.sum(real_list) != 0:
                    sns.distplot(diff_list, color='green', rug=True, hist=False)

                plt.subplot(3, 3, (8,9))
                plt.tight_layout()
                plt.axis('off')
                acceptance_rate = (sum(i <= 0.20 for i in diff_list) / len(real_list)) * 100
                # diff_normality_test
                if len(diff_list) >2:
                    p, stat = shapiro(diff_list)
                    if p > 0.5:
                        plt.text(0.1, 0.3, "Differenc follows a normal distribution \n"
                                           " with coefficient of varaition of: " + str(round(variation(diff_list),2)))
                    else:
                        plt.text(0.1, 0.3, "Differenc is not a normal distribution \n"
                                           " mean and standard deviation of: " + str(round(variation(diff_list),2)))

                if len(np.nonzero(real_list)) >0 and len(np.nonzero(simulation_list))>0:
                    normal_real_list = (real_list - np.min(real_list))/(np.max(real_list)-np.min(real_list))
                    normal_simulation_list = (simulation_list - np.min(simulation_list)) / (np.max(simulation_list) - np.min(simulation_list))
                    normal_diff_list = abs(np.array(normal_real_list) - np.array(normal_simulation_list))
                val_image_names.append(vname)
                plt.subplots_adjust(wspace=0,hspace=1)
                plt.savefig('static/images/'+str(vname)+'_validation.png')

            return val_image_names
