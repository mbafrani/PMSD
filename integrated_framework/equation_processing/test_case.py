# ##### test case ####
# mean_values = {
#     'arrival_rate1w': 594.5094339622641,
#     'finish_rate1w': 575.9433962264151,
#     'num_of_unique_resource1w': 73.13207547169812,
#     'process_active_time1w': 311763.0443557967,
#     'service_time_per_case1w': 51.25092766369817,
#     'time_in_process_per_case1w': 523.4869462001786,
#     'waiting_time_in_process_per_case1w': 472.2360185364805,
#     'num_in_process_case1w': 1837.9811320754718}
#
#
# all_best_equations = {
#     'arrival_rate1w': [[-0.00457528, -0.5282063, 0.00182478, -0.45473603, -0.6990708,
#                         -0.24433476, 0.00516066], 562.8335836178192,
#                        ['finish_rate1w', 'num_of_unique_resource1w', 'process_active_time1w', 'service_time_per_case1w',
#                         'time_in_process_per_case1w', 'waiting_time_in_process_per_case1w', 'num_in_process_case1w'], 'Ridge Regression'],
#     'process_active_time1w': [[2.56348692e-11, 3.06184582e-12, -3.99453330e-11, -8.64875084e-12,
#                                2.75361996e-11, 3.61845601e-11, -7.93523367e-12, 4.89327596e-14,
#                                6.88438291e-15, -2.32588303e-13, 3.33333333e-01, 6.66666667e-01,
#                                3.33333333e-01, -2.75707383e-14, -3.04942913e-15, 5.29591895e-14,
#                                2.51146825e-14, -2.10074758e-14, -4.76751993e-15, 4.30243610e-15,
#                                3.37980938e-13, 3.04408907e-13, 3.53426972e-14, -2.72673533e-13,
#                                8.56546670e-14, 2.76211230e-13, 7.03591936e-14, -2.05650391e-13,
#                                2.46775968e-14, -2.01410070e-14, -1.11766306e-13, 5.85701456e-14,
#                                8.19791283e-14, -3.70404662e-14, 1.89084859e-16], 0.016214378352742642,
#                               [[('arrival_rate1w', 1)], [('finish_rate1w', 1)], [('num_of_unique_resource1w', 1)],
#                                [('service_time_per_case1w', 1)], [('time_in_process_per_case1w', 1)],
#                                [('waiting_time_in_process_per_case1w', 1)], [('num_in_process_case1w', 1)], [('arrival_rate1w', 2)],
#                                [('arrival_rate1w', 1), ('finish_rate1w', 1)], [('arrival_rate1w', 1), ('num_of_unique_resource1w', 1)],
#                                [('arrival_rate1w', 1), ('service_time_per_case1w', 1)],
#                                [('arrival_rate1w', 1), ('time_in_process_per_case1w', 1)],
#                                [('arrival_rate1w', 1), ('waiting_time_in_process_per_case1w', 1)],
#                                [('arrival_rate1w', 1), ('num_in_process_case1w', 1)], [('finish_rate1w', 2)],
#                                [('finish_rate1w', 1), ('num_of_unique_resource1w', 1)],
#                                [('finish_rate1w', 1), ('service_time_per_case1w', 1)],
#                                [('finish_rate1w', 1), ('time_in_process_per_case1w', 1)],
#                                [('finish_rate1w', 1), ('waiting_time_in_process_per_case1w', 1)],
#                                [('finish_rate1w', 1), ('num_in_process_case1w', 1)], [('num_of_unique_resource1w', 2)],
#                                [('num_of_unique_resource1w', 1), ('service_time_per_case1w', 1)],
#                                [('num_of_unique_resource1w', 1), ('time_in_process_per_case1w', 1)],
#                                [('num_of_unique_resource1w', 1), ('waiting_time_in_process_per_case1w', 1)],
#                                [('num_of_unique_resource1w', 1), ('num_in_process_case1w', 1)], [('service_time_per_case1w', 2)],
#                                [('service_time_per_case1w', 1), ('time_in_process_per_case1w', 1)],
#                                [('service_time_per_case1w', 1), ('waiting_time_in_process_per_case1w', 1)],
#                                [('service_time_per_case1w', 1), ('num_in_process_case1w', 1)], [('time_in_process_per_case1w', 2)],
#                                [('time_in_process_per_case1w', 1), ('waiting_time_in_process_per_case1w', 1)],
#                                [('time_in_process_per_case1w', 1), ('num_in_process_case1w', 1)],
#                                [('waiting_time_in_process_per_case1w', 2)],
#                                [('waiting_time_in_process_per_case1w', 1), ('num_in_process_case1w', 1)],
#                                [('num_in_process_case1w', 2)]], 'polynomial regression']}
#
#
# output = pre_processing_equations(mean_values, all_best_equations)
#
# print(type(output))
#
# import  pandas as pd
# sf_vars = {'num_in_process_case1w':[['arrival_rate1w'], ['finish rate1w']]}
# path = '/Users/robin/GitLab/thesis/data/Active_1W_sdlog.csv'
# sd = pd.read_csv(path)
#
# features = sd.columns
#
# temp = {k:k.replace(' ', '_').lower() for k in features}
# print(temp)
# sd.rename(columns=temp, inplace = True)
#
# print(sd)
#
# process_equations, stock_lines, lines_biger, lines_samller = process_equations_new(sd, output, sf_vars)
#
# print('updated, process equations', process_equations)
# print('stock lines', stock_lines)
# print('lines bigger', lines_biger)
# print('lines smaller', lines_samller)


# results

# updated, process equations :{'num_in_process_case1w': [(1, 'arrival_rate1w'), (-1, 'finish rate1w'), 632.0], 'arrival_rate1w': [('num_of_unique_resource1w', 'arrival_rate1w', -0.5282063), ('service_time_per_case1w', 'arrival_rate1w', -0.45473603), ('time_in_process_per_case1w', 'arrival_rate1w', -0.6990708), ('waiting_time_in_process_per_case1w', 'arrival_rate1w', -0.24433476), ('num_in_process_case1w', 'arrival_rate1w', 0.00516066), (1129.0974493955032, 'Ridge Regression')], 'process_active_time1w': [('arrival_rate1w', 'process_active_time1w', 0.333333333), ('service_time_per_case1w', 'process_active_time1w', 0.333333333), ('arrival_rate1w', 'process_active_time1w', 0.666666667), ('time_in_process_per_case1w', 'process_active_time1w', 0.666666667), ('arrival_rate1w', 'process_active_time1w', 0.333333333), ('waiting_time_in_process_per_case1w', 'process_active_time1w', 0.333333333), (0.016214403541412847, 'polynomial regression')]}
# stock lines {('arrival_rate1w', 'num_in_process_case1w', 1), ('finish rate1w', 'num_in_process_case1w', -1)}
# lines bigger {('waiting_time_in_process_per_case1w', 'process_active_time1w', 0.333333333), ('service_time_per_case1w', 'process_active_time1w', 0.333333333), ('time_in_process_per_case1w', 'process_active_time1w', 0.666666667), ('num_in_process_case1w', 'arrival_rate1w', 0.00516066), ('service_time_per_case1w', 'arrival_rate1w', -0.45473603), ('time_in_process_per_case1w', 'arrival_rate1w', -0.6990708), ('arrival_rate1w', 'process_active_time1w', 0.333333333), ('arrival_rate1w', 'process_active_time1w', 0.666666667), ('num_of_unique_resource1w', 'arrival_rate1w', -0.5282063), ('waiting_time_in_process_per_case1w', 'arrival_rate1w', -0.24433476)}
# lines smaller {('finish_rate1w', 'arrival_rate1w', -0.00457528), ('process_active_time1w', 'arrival_rate1w', 0.00182478)}