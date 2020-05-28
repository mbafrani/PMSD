import csv
import re
import scipy
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import Counter
from scipy.stats import variation,ks_2samp,shapiro
from pm4py.objects.conversion.log import factory as log_converter
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.xes import factory as xes_importer



class Complete_sd:

    def get_input_file(self,event_log_address):

        log_format = event_log_address.split('.')[-1]

        if str(log_format) == 'csv':
            event_log = pd.read_csv(event_log_address)

        elif str(log_format) == 'xes':
            xes_log = xes_importer.import_log(event_log_address)
            event_log = log_converter.apply(xes_log)
            csv_exporter.export_log(event_log, "event_log_repaired.csv")
            event_log = pd.read_csv("event_log_repaired.csv")

        event_log_attributes = event_log.columns

        return event_log,event_log_attributes

    def add_needed_column(self, event_log, event_log_attributes):

        CaseID = event_log_attributes[0]
        Activity = event_log_attributes[1]
        Resource = event_log_attributes[2]
        StartTimestamp = event_log_attributes[3]
        CompleteTimestamp = event_log_attributes[4]

        if str(StartTimestamp) != str(CompleteTimestamp):
            event_log = event_log.rename(index = str, columns={CaseID: "Case ID",Activity:'Activity',
                                   Resource:'Resource',StartTimestamp:'Start Timestamp',
                                   CompleteTimestamp:'Complete Timestamp'})
        else:
            event_log = event_log.rename(index=str, columns={CaseID: "Case ID", Activity: 'Activity',
                                                            Resource: 'Resource', StartTimestamp: 'Start Timestamp'})
            event_log['Complete Timestamp']=event_log['Start Timestamp']

        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'],errors = 'coerce')
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'],errors = 'coerce')
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration
        event_log['Activity'] = event_log['Activity'].str.replace(" ", "")


        return event_log

     #TODO: General Calculation for General Same as Time Windows -series NEW

    def TW_discovery_process_calculation_twlist(self, tw_lists,event_log, aspect):
        Overall_dict = {}
        Arrival_rate_dict = {}
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'], errors='coerce')
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'], errors='coerce')
        event_log['Activity Duration'] = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        #start_unit_log = min(event_log['Start Timestamp'])
        #end_log = max(event_log['Complete Timestamp'])
        arr_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        fin_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        case_dur_temp_log = event_log.groupby(['Case ID'])
        Name_General_selected_variables_dict = []
        for tw_list in tw_lists:
            # todo Arrival Rate of Cases per Day, Week, Month
            y = int((re.findall(r'\d+', tw_list))[0])
            z = tw_list[-1]

            if z == "H" and (24 % y == 0) or z != 'H':

                startdiff = []
                startforeach = []
                for actname, actgroup in arr_temp_event_log:
                    startforeach.append(actgroup['Start Timestamp'].values[0])
                startforeach.sort()

                for i in range(len(startforeach) - 1):
                    startdiff.append((startforeach[i + 1] - startforeach[i]))

                count_sta = Counter(startforeach)
                df_startforeach = pd.Series(count_sta).to_frame()
                Hourly = df_startforeach.resample(str(tw_list)).sum()
                Hourly.columns = ['hourly']

                # todo Finish Rate of Cases per Day, Week, Month
                enddiff = []
                endforeach = []
                for actname, actgroup in fin_temp_event_log:
                    endforeach.append(actgroup['Complete Timestamp'].values[-1])
                endforeach.sort()

                for i in range(len(endforeach) - 1):
                    enddiff.append((endforeach[i + 1] - endforeach[i]))

                enddiff = pd.to_timedelta(enddiff).seconds / 3600
                counter_intervals = Counter(enddiff)
                ecount_sta = Counter(endforeach)
                edf_startforeach = pd.Series(ecount_sta).to_frame()
                eHourly = edf_startforeach.resample(str(tw_list)).sum()

                eHourly.columns = ['ehourly']

                # Todo Compute the number of in process cases (MAX Capacity)
                Hourly_df = pd.concat([Hourly, eHourly], axis=1, join='outer')
                Hourly_df.fillna(0, inplace=True)

                h_case_in_process = (Hourly_df['hourly'] - Hourly_df['ehourly'])
                temp_list_inproc = h_case_in_process.tolist()
                for i in range(len(temp_list_inproc)):
                    if i == 0:
                        temp_list_inproc[i] = h_case_in_process.tolist()[i]
                    else:
                        temp_list_inproc[i] = temp_list_inproc[i] + temp_list_inproc[i - 1]

                max_h_case_in_process = max(h_case_in_process)

            # TODO Process whole active Time and Process active time per W,D,M
            try:
                process_active_time = event_log['Activity Duration'].sum()
            except ValueError as err:
                print(err)

            temp_active_time = event_log[['Start Timestamp', 'Activity Duration']]
            temp_active_time['Start Timestamp'] = pd.to_datetime(temp_active_time['Start Timestamp'])
            sort_start_duration = temp_active_time.sort_values('Start Timestamp')
            sort_start_duration.set_index('Start Timestamp', inplace=True)
            process_H_active_time_df = sort_start_duration.resample(str(tw_list)).sum()
            process_H_active_time = process_H_active_time_df['Activity Duration'].values
            process_H_active_time = pd.to_timedelta(process_H_active_time).total_seconds() / 3600
            # TODO List of time in process per case and real time service (Case Durations)
            case_duration_list = []
            case_real_duration_list = []
            case_real_duration_list_waiting = []
            case_duration_list_waiting = []
            case_duration_dict = {}
            case_real_duration_dict = {}
            for dcase, dgroup in case_dur_temp_log:
                case_real_duration_list_waiting.append(np.sum(dgroup['Activity Duration']))
                case_duration_list_waiting.append(
                    np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp']))
                case_duration_dict[np.min(dgroup['Start Timestamp'])] = pd.to_timedelta(
                    np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp'])).total_seconds() / 3600
                case_real_duration_dict[np.min(dgroup['Start Timestamp'])] = pd.to_timedelta(
                    np.sum(dgroup['Activity Duration'])).total_seconds() / 3600

            case_duration_list = list(case_duration_dict.values())
            case_real_duration_list = list(case_real_duration_dict.values())

            case_duration_df = pd.DataFrame(case_duration_dict.items(), columns=['Start Timestamp', 'Case Duration'])
            case_duration_df = case_duration_df.sort_values('Start Timestamp')
            case_duration_df.set_index('Start Timestamp', inplace=True)
            case_duration_H_df = case_duration_df.resample(str(tw_list)).sum()
            case_duration_H_df.insert(0, 'Avg Case Duration', case_duration_df.resample(str(tw_list)).mean())
            case_duration_H_df = case_duration_H_df.fillna(0)
            case_real_duration_df = pd.DataFrame(case_real_duration_dict.items(),
                                                 columns=['Start Timestamp', 'Case Duration'])
            case_real_duration_df = case_real_duration_df.sort_values('Start Timestamp')
            case_real_duration_df.set_index('Start Timestamp', inplace=True)
            case_real_duration_H_df = case_real_duration_df.resample(str(tw_list)).sum()
            case_real_duration_H_df.insert(0, 'Avg Case Duration', case_real_duration_df.resample(str(tw_list)).mean())
            case_real_duration_H_df=case_real_duration_H_df.fillna(0)

            case_duration_list_waiting = pd.to_timedelta(case_duration_list_waiting).total_seconds() / 3600
            case_real_duration_list_waiting = pd.to_timedelta(case_real_duration_list_waiting).total_seconds() / 3600
            waiting_time = np.array(case_duration_list_waiting) - np.array(case_real_duration_list_waiting)

            case_duration_coutner = Counter(case_duration_list)
            case_real_duration_coutner = Counter(case_real_duration_list)
            avg_case_duration = np.mean(case_duration_list)
            avg_case_real_duration = np.mean(case_real_duration_list)

            # TODO Number of Resources and unique resources per
            temp_event_log_start_index = event_log.set_index('Start Timestamp')
            temp_group_h = temp_event_log_start_index.groupby(pd.Grouper(freq=str(tw_list)))
            num_unique_resource_h = (temp_group_h['Resource'].nunique()).values


            # TODO Number of Resources and unique resources per case
            uniqe_resource_list_per_case = []
            resource_list_per_case = []
            for rc, rgroup in case_dur_temp_log:
                resource_per_case = rgroup['Resource']
                unique_resource_per_case = np.unique(resource_per_case)
                uniqe_resource_list_per_case.append(len(unique_resource_per_case))
                resource_list_per_case.append(len(resource_per_case))
            average_waiting = pd.array(case_duration_H_df['Avg Case Duration'].tolist()) - pd.array(
                case_real_duration_H_df['Avg Case Duration'].tolist())
            average_waiting = np.array(average_waiting).tolist()
            average_waiting = [0 if x < 0 else x for x in average_waiting]
            # TODO Create Overall Dict
            Name_General_selected_variables=(str(aspect) + "_sdlog.csv")
            General_selected_variables_dict = {"Arrival rate" + str(tw_list): Hourly['hourly'].values.tolist(),
                                               "Finish rate" + str(tw_list): (eHourly['ehourly'].values).tolist(),
                                               "Num of unique resource" + str(tw_list): num_unique_resource_h.tolist(),
                                               "Process active time" + str(tw_list): case_duration_H_df[
                                                   'Case Duration'].tolist(),
                                               "Service time per case" + str(tw_list): case_real_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Time in process per case" + str(tw_list): case_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Waiting time in process per case" + str(tw_list): average_waiting,
                                               "Num in process case" + str(tw_list): temp_list_inproc,
                                               }
            for k,v in General_selected_variables_dict.items():
                General_selected_variables_dict[k]=pd.Series(General_selected_variables_dict[k], dtype=object).fillna(0).tolist()
            with open(r"Outputs/"+str(aspect) + "_sdlog.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(General_selected_variables_dict.keys())
                x = zip(*General_selected_variables_dict.values())
                xx = zip(*x)
                xxx = zip(*xx)
                writer.writerows(xxx)
                # TODO Test data and Distribution
                vid = 1
                plt.figure()
                for dk, dv in General_selected_variables_dict.items():

                    # plt.hist(dv)
                    try:
                        mean_data = round(np.mean(dv), 2)
                        var_data = round(np.var(dv), 2)
                        std_data = round(np.std(dv), 2)
                        min_data = round(min(dv), 2)
                        max_data = round(max(dv), 2)
                    except:
                        mean_data = 0
                        var_data = 0
                        std_data = 0
                        min_data = min(dv)
                        max_data = max(dv)
                    cont_dist_names = ['uniform', 'expon', 'norm', 'pareto', 'gamma']
                    dis_dist_names = ['poisson']
                    params = {}
                    dist_results = []

                    D, p = scipy.stats.kstest(dv, 'poisson', args=(mean_data,))
                    dist_results.append(('poisson', p))
                    for dist_name in cont_dist_names:
                        dist = getattr(scipy.stats, dist_name)
                        param = dist.fit(dv)
                        params[dist_name] = param
                        # Kolmogorov-Smirnov test for goodness of fit. D shows the distance between two samples the lower the more similarity.
                        D, p = scipy.stats.kstest(dv, dist_name, args=param)
                        dist_results.append((dist_name, p))

                    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
                    # sns.distplot(dv)
                    plt.rcParams["figure.figsize"] = [10, 10]
                    plt.subplot(3, 3, vid)
                    plt.tight_layout()
                    plt.gca().set_title(str(dk), size=8)
                    plt.axis('off')
                    # sns.distplot(dv)
                    plt.text(0.1, 0.2, str(best_dist) + '\n' + 'best P:' + str(round(best_p, 2)) + '\n' + 'Mean:' + str(
                        round(mean_data, 2)) +
                             '\n' + 'STD:' + str(round(std_data, 2)) +
                             '\n' + 'Min:' + str(min_data) + '\n' + 'Max:' + str(
                        max_data) + '\n' + 'Coefficient of Variance:' + str(np.round(variation(dv), 2)),size=8)
                    vid += 1

                mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())
                plt.rcParams["figure.figsize"] = [20, 20]
                plt.resize(100, 100)
                plt.savefig('static/images/' + str(aspect) + '_sdlog.csv.png', dpi=100)
               #plt.show()

        return  Name_General_selected_variables

    #TODO: Removing Inactive with inactive check same as TIme window series
    def Post_process_tw(self,SD_Log,aspect):

        SD_Log = pd.read_csv(SD_Log)
        SD_Log = SD_Log.fillna(0)
        target_feature_values = SD_Log[SD_Log.columns[0]]
        new_df = SD_Log.loc[target_feature_values == 0]
        inactive_array = (new_df == 0).astype(int).sum(axis=1) / len(new_df.columns)
        my_list = list(inactive_array[inactive_array < 0.9].index)
        newnewdf = new_df.loc[my_list]
        ind = new_df.index
        Active_SD_Log = SD_Log.drop(SD_Log.index[ind])
        diff = list()
        for i in range(1, len(ind)):
            value = ind[i] - ind[i - 1]
            diff.append(value)
        # plt.bar([x for x in range(0,len(diff))],diff)
        Active_SD_Log.to_csv(r"Outputs\\"+(str(aspect) + "_sdlog.csv"),index=False)
        return (str(aspect) + "_sdlog.csv")
