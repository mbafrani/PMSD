import csv
import re
import os
import scipy
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
from collections import Counter
from scipy.stats import variation,ks_2samp,shapiro
from pm4py.objects.conversion.log import converter as log_converter
#from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer



class Complete_sd:

    def get_input_file(self,event_log_address):

        log_format = event_log_address.split('.')[-1]

        if str(log_format) == 'csv':
            event_log = pd.read_csv(event_log_address)

        elif str(log_format) == 'xes':
            xes_log = xes_importer.apply(event_log_address)
            event_log = log_converter.apply(xes_log, variant=log_converter.Variants.TO_DATA_FRAME)
            event_log.to_csv("event_log_repaired.csv")
            #csv_exporter.export_log(event_log, "event_log_repaired.csv")
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

        event_log['Activity']=event_log['Activity'].astype(str)
        event_log['Resource']=event_log['Resource'].astype(str)
        event_log['Activity'] = event_log['Activity'].str.replace(" ", "")
        event_log['Activity'] = event_log['Activity'].str.replace("_", "")
        event_log['Resource'] = event_log['Resource'].str.replace(" ", "")
        event_log['Resource'] = event_log['Resource'].str.replace("_", "")


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
                case_duration_list_waiting.append(np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp']))
                case_duration_dict[np.min(dgroup['Start Timestamp'])] = pd.to_timedelta(np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp'])).total_seconds() / 3600
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
            num_unique_act_h = (temp_group_h['Activity'].nunique()).values

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
            General_selected_variables_dict = {str(aspect)+"_Arrival rate" + str(tw_list): Hourly['hourly'].values.tolist(),
                                               str(aspect)+"_Finish rate" + str(tw_list): (eHourly['ehourly'].values).tolist(),
                                               str(aspect) + "_Num of unique resource" + str(tw_list): num_unique_resource_h.tolist(),
                                               str(aspect) + "_Process active time" + str(tw_list): case_duration_H_df["Case Duration"].tolist(),
                                               str(aspect) + "_Service time per case" + str(tw_list): case_real_duration_H_df[ 'Avg Case Duration'].tolist(),
                                               str(aspect) +  "_Time in process per case" + str(tw_list): case_duration_H_df['Avg Case Duration'].tolist(),
                                               str(aspect) +  "_Waiting time in process per case" + str(tw_list): average_waiting,
                                               str(aspect) +   "_Num in process case" + str(tw_list): temp_list_inproc,
                                               str(aspect) + "_Num of unique activitis"+str(tw_list): num_unique_act_h.tolist()
                                               }
            for k,v in General_selected_variables_dict.items():
                General_selected_variables_dict[k]=pd.Series(General_selected_variables_dict[k], dtype=object).fillna(0).tolist()
            if len(aspect) >10:
                aspect=aspect[:10]
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
                        try:
                            param = dist.fit(dv)
                            params[dist_name] = param
                        # Kolmogorov-Smirnov test for goodness of fit. D shows the distance between two samples the lower the more similarity.
                            D, p = scipy.stats.kstest(dv, dist_name, args=param)
                            dist_results.append((dist_name, p))
                        except:
                            pass

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

        return  Name_General_selected_variables,General_selected_variables_dict

    #TODO: Removing Inactive with inactive check same as TIme window series
    def Post_process_tw(self,SD_Log,aspect):

        SD_Log =pd.DataFrame(SD_Log.items())
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
        outputpath=os.path.join("Outputs",(+str(aspect) + "_sdlog.csv"))
        Active_SD_Log.to_csv(outputpath,columns=list(Active_SD_Log.columns),index=False)
        return (str(aspect) + "_sdlog.csv")

    #TODO: Per Resrouce Calculation Specifically for Waiting time with differeten time duraiton
    def break_log_for_res(self, time_unit, event_log):
        start_unit_log = min(event_log['Start Timestamp'])
        start_unit_log = pd.to_datetime(start_unit_log)
        start_unit_log = pd.to_datetime(start_unit_log.replace(hour=00, minute=00))
        end_log = max(event_log['Complete Timestamp'])
        y = int((re.findall(r'\d+', str(time_unit)))[0])
        z = time_unit[-1]
        if "H" in str(time_unit)[-3]:
            time_unit = "60"
        elif "D" in str(time_unit)[-3]:
            time_unit = "1440"
        elif "W" in str(time_unit)[-3]:
            time_unit = "10080"
        elif "M" in str(time_unit)[-3]:
            time_unit = '40320'

        minutes = y * int(str(time_unit))
        time_unit = pd.DateOffset(minutes=minutes)
        end_log = pd.to_datetime(end_log)
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            event_log_unit['Start Timestamp'] = pd.to_datetime(event_log_unit['Start Timestamp'])
            event_log_unit['Complete Timestamp'] = pd.to_datetime(event_log_unit['Complete Timestamp'])
            if 'Event Duration' not in event_log_unit.columns:
                event_log_unit['Event Duration'] = event_log_unit['Complete Timestamp'] - event_log_unit[
                    'Start Timestamp']
            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Resource'])
            case_temp_event_log = event_log_unit.groupby(['Case ID'])
            act_num_speed_dict = defaultdict(list)
            casegroup_act_dict = defaultdict(list)
            casegroup_arrival_act_dict = defaultdict(list)

            for casename, casegroup in case_temp_event_log:
                casegroup = casegroup.sort_values(['Start Timestamp'])
                casegroup_act_list = casegroup['Resource'].unique()
                casegroup_start_list = casegroup['Start Timestamp'].values
                casegroup_comp_list = casegroup['Complete Timestamp'].values

                for caseact in casegroup_act_list:
                    ix0 = casegroup_act_list.tolist().index(caseact)
                    ix1 = ix0 + 1

                    # if len(casegroup_act_list) == 1:
                    #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        if ix0 == 0:
                            casegroup_act_dict[casegroup_act_list[ix0]].append(0)
                        case_act_diff = pd.to_timedelta(casegroup_comp_list[ix1] - casegroup_start_list[ix0]).seconds / 3600
                        casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                        if (casegroup_comp_list[ix0]) <= (casegroup_start_list[ix1]):
                            casegroup_act_dict[casegroup_act_list[ix1]].append(case_act_diff)
                        else:
                            casegroup_act_dict[casegroup_act_list[ix1]].append(0)

            for actname, actgroup in temp_event_log:
                list_case_per_act = []
                list_case_per_act.append(actgroup['Case ID'].nunique())

                list_complete = actgroup['Complete Timestamp'].values
                list_start = actgroup['Start Timestamp'].values
                list_act_idle_time = []
                list_act_waiting = []

                for i in range(len(list_complete) - 1):
                    if pd.to_timedelta(list_start[i + 1] - list_complete[i]).seconds < 0:
                        list_act_waiting.append(abs(list_start[i + 1] - list_complete[i]))
                        list_act_idle_time.append(pd.to_timedelta(0))
                    else:
                        list_act_waiting.append(pd.to_timedelta(0))
                        list_act_idle_time.append(list_start[i + 1] - list_complete[i])
                if len(list_act_idle_time) == 0:
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting) == 0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_dur_event_list = pd.to_timedelta(act_dur_event_list).values
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                # 0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1:
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                # act_num_speed_dict[actname].append(len(act_num_event_list))

                # 1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))

                # 2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))

                # 3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))
                            # 4 whole waiting time
                            act_num_speed_dict[actname].append((np.sum(casegroup_act_dict[caseidact])))

                # 4 whole waiting time
                # act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                # 5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                # 6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                # 7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                # 8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)

            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit

        return unit_time_act_num_dur_dict

    # TODO: Per Activity Calculation Specifically for Waiting time with differeten time duraiton
    def break_log_for_act(self, time_unit, event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        start_unit_log = pd.to_datetime(start_unit_log)
        start_unit_log = pd.to_datetime(start_unit_log.replace(hour=00, minute=00))
        end_log = max(event_log['Complete Timestamp'])
        y = int((re.findall(r'\d+', str(time_unit)))[0])
        z = time_unit[-1]
        if "H" in str(time_unit)[-3]:
            time_unit= "60"
        elif "D" in str(time_unit)[-3]:
            time_unit ="1440"
        elif "W" in str(time_unit)[-3]:
            time_unit="10080"
        elif "M" in str(time_unit)[-3]:
            time_unit='40320'

        minutes = y*int(str(time_unit))
        time_unit = pd.DateOffset(minutes=minutes)
        #  event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        #  grouped_log_date = event_log.groupby(by=event_log['Start Timestamp'].dt.date)
        end_log = pd.to_datetime(end_log)
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)
        end_unit_log = start_unit_log + time_unit

        while end_unit_log <= (end_log - time_unit):
            event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            event_log_unit['Start Timestamp']=pd.to_datetime(event_log_unit['Start Timestamp'])
            event_log_unit['Complete Timestamp'] = pd.to_datetime(event_log_unit['Complete Timestamp'])
            if 'Event Duration' not in event_log_unit.columns:
                event_log_unit['Event Duration'] = event_log_unit['Complete Timestamp'] - event_log_unit['Start Timestamp']

            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Activity'])
            case_temp_event_log = event_log_unit.groupby(['Case ID'])
            act_num_speed_dict = defaultdict(list)
            casegroup_act_dict = defaultdict(list)
            casegroup_arrival_act_dict = defaultdict(list)

            for casename, casegroup in case_temp_event_log:
                casegroup = casegroup.sort_values(['Start Timestamp'])
                casegroup_act_list = casegroup['Activity'].values
                casegroup_start_list = casegroup['Start Timestamp'].values
                casegroup_comp_list = casegroup['Complete Timestamp'].values

                for caseact in casegroup_act_list:
                    ix0 = casegroup_act_list.tolist().index(caseact)
                    ix1 = ix0 + 1

                    # if len(casegroup_act_list) == 1:
                    #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        if ix0==0:
                            casegroup_act_dict[casegroup_act_list[ix0]].append(0)
                        case_act_diff = pd.to_timedelta(
                        casegroup_start_list[ix1] - casegroup_comp_list[ix0]).seconds / 3600
                        casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                        if (casegroup_comp_list[ix0]) <= (casegroup_start_list[ix1]):
                            casegroup_act_dict[casegroup_act_list[ix1]].append(case_act_diff)
                        else:
                            casegroup_act_dict[casegroup_act_list[ix1]].append(0)

            for actname, actgroup in temp_event_log:
                list_case_per_act = []
                list_case_per_act.append(actgroup['Case ID'].nunique())

                list_complete = actgroup['Complete Timestamp'].values
                list_start = actgroup['Start Timestamp'].values
                list_act_idle_time = []
                list_act_waiting = []

                for i in range(len(list_complete) - 1):
                    if pd.to_timedelta(list_start[i + 1] - list_complete[i]).seconds < 0:
                        list_act_waiting.append(abs(list_start[i + 1] - list_complete[i]))
                        list_act_idle_time.append(pd.to_timedelta(0))
                    else:
                        list_act_waiting.append(pd.to_timedelta(0))
                        list_act_idle_time.append(list_start[i + 1] - list_complete[i])
                if len(list_act_idle_time) == 0:
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting) == 0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']

                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_dur_event_list= pd.to_timedelta(act_dur_event_list)
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                # 0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1:
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                # act_num_speed_dict[actname].append(len(act_num_event_list))

                # 1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list.values)))

                # 2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list.values)))

                # 3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))
                            # 4 whole waiting time
                            act_num_speed_dict[actname].append((np.sum(casegroup_act_dict[caseidact])))

                # 4 whole waiting time
                # act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                # 5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                # 6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                # 7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                # 8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)
                # 9 number of unique resource
                act_num_speed_dict[actname].append(actgroup['Resource'].nunique())
                # 10 number of involved resource
                act_num_speed_dict[actname].append(len(actgroup['Resource']))

            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit

        return unit_time_act_num_dur_dict

    #TODO: Create and calculate the values for activity and resource view (single one) after breaking logs
    def create_features_name(self,view_list,unique_act_res):

        #unique_act = event_log['Activity'].unique()
        #unique_res = event_log['Resource'].unique()


        features_list = []
        features_name_list=[]
        features_name_list.append(unique_act_res)
        if str(view_list) == 'Activities':

            for ac in features_name_list:
                if ac in features_name_list:
                    features_list.append( 'act' + '_' + str(ac) + '_avg_arrival')
                    features_list.append( 'act' + '_' +str(ac) + '_avg_duration')
                    features_list.append( 'act' + '_' +str(ac) + '_whole_duration')
                    features_list.append( 'act' + '_' +str(ac) + '_avgwaiting_time')
                    features_list.append( 'act' + '_' +str(ac) + '_wholewaiting_time')
                    features_list.append( 'act' + '_' +str(ac) + '_waiting_events')
                    features_list.append( 'act' + '_' +str(ac) + '_finished_events')
                    features_list.append('act' + '_' +str(ac) + '_idle_time')
                    features_list.append('act' + '_' + str(ac) + '_inprocess_events')
                    features_list.append( 'act' + '_' +str(ac) + '_unique_resources')
                    features_list.append( 'act' + '_' +str(ac) + '_engaged_resources')

        if str(view_list) == 'Resources':
            for rs in features_name_list:

                    features_list.append('res' + '_' + str(rs) + '_avg_arrival')
                    features_list.append('res' + '_' + str(rs) + '_avg_duration')
                    features_list.append('res' + '_' + str(rs) + '_whole_duration')
                    features_list.append('res' + '_' + str(rs) + '_avgwaiting_time')
                    features_list.append('res' + '_' + str(rs) + '_wholewaiting_time')
                    features_list.append('res' + '_' + str(rs) + '_waiting_events')
                    features_list.append('res' + '_' + str(rs) + '_finished_events')
                    features_list.append('res' + '_' + str(rs) + '_idle_time')
                    features_list.append('res' + '_' + str(rs) + '_inprocess_events')

        return features_list

    def select_features(self, features_list, event_log, time_unit,unique_act_res):
        fvalues_dict = {}
        unit_time_act_num_dur_dict = {}
        unit_time_res_num_dur_dict =  {}
        if unique_act_res =='Activities':
            unit_time_act_num_dur_dict = self.break_log_for_act(time_unit, event_log)
        elif unique_act_res == 'Resources':
            unit_time_res_num_dur_dict = self.break_log_for_res(time_unit, event_log)
        for f in features_list:
            fview = f.split('_')[0]
            fname_list = f.split('_')[1:-2]
            fname = ''
            if len(fname_list) > 1:
                for w in fname_list:
                    if w != fname_list[-1]:
                        fname = fname + w + '_'
                    else:
                        fname = fname + w
            else:
                for w in fname_list:
                    fname = fname + w
            ftype_list = f.split('_')[-2:]
            ftype = ''
            for t in ftype_list:
                ftype = ftype + t
            fvalues = []

            if fview == 'res':
                for t, values in unit_time_res_num_dur_dict.items():
                    if len(values) != 0 and fname in unit_time_res_num_dur_dict.get(t):
                        if str(ftype) == 'avgarrival':
                            fvalues.append(values.get(fname)[0])
                        elif ftype == 'avgduration':
                            fvalues.append(values.get(fname)[1])
                        elif ftype == 'wholeduration':
                            fvalues.append(values.get(fname)[2])
                        elif ftype == 'avgwaitingtime':
                            fvalues.append(values.get(fname)[3])
                        elif ftype == 'wholewaitingtime':
                            fvalues.append(values.get(fname)[4])
                        elif ftype == 'waitingevents':
                            fvalues.append(values.get(fname)[5])
                        elif ftype == 'finishedevents':
                            fvalues.append(values.get(fname)[6])
                        elif ftype == 'idletime':
                            fvalues.append(values.get(fname)[7])
                        elif ftype == 'inprocessevents':
                            fvalues.append(values.get(fname)[8])
                if len(fvalues) != 0 and isinstance(fvalues[0], pd.Timedelta):
                    fvalues = pd.to_timedelta(fvalues).seconds / 3600
                fvalues_dict.update({f: fvalues})

            if fview == 'act':
                for t, values in unit_time_act_num_dur_dict.items():
                    if len(values) != 0 and fname in unit_time_act_num_dur_dict.get(t):
                        if str(ftype) == 'avgarrival':
                            fvalues.append(values.get(fname)[0])
                        elif ftype == 'avgduration':
                            fvalues.append(values.get(fname)[1])
                        elif ftype == 'wholeduration':
                            fvalues.append(values.get(fname)[2])
                        elif ftype == 'avgwaitingtime':
                            fvalues.append(values.get(fname)[3])
                        elif ftype == 'wholewaitingtime':
                            fvalues.append(values.get(fname)[4])
                        elif ftype == 'waitingevents':
                            fvalues.append(values.get(fname)[5])
                        elif ftype == 'finishedevents':
                            fvalues.append(values.get(fname)[6])
                        elif ftype == 'idletime':
                            fvalues.append(values.get(fname)[7])
                        elif ftype == 'inprocessevents':
                            fvalues.append(values.get(fname)[8])
                        elif ftype == 'uniqueresources':
                            fvalues.append(values.get(fname)[9])
                        elif ftype == 'engagedresources':
                            fvalues.append(values.get(fname)[10])


                if len(fvalues) != 0 and isinstance(fvalues[0], pd.Timedelta):
                    fvalues = pd.to_timedelta(fvalues).seconds / 3600

                fvalues_dict.update({f: fvalues})

        for k, v in fvalues_dict.items():
            fvalues_dict[k] = pd.Series(fvalues_dict[k], dtype=object).fillna(
                0).tolist()
        Name_General_selected_variables = (str(fname) + "_sdlog.csv")
        with open(r"Outputs/" + str(fname) + "_sdlog.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fvalues_dict.keys())
            x = zip(*fvalues_dict.values())
            xx = zip(*x)
            xxx = zip(*xx)
            writer.writerows(xxx)
            # TODO Test data and Distribution
            vid = 1
            plt.figure()
            for dk, dv in fvalues_dict.items():

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
                    #min_data = min(dv)
                    min_data = 0
                    #max_data = max(dv)
                    max_data = 0
                cont_dist_names = ['uniform', 'expon', 'norm', 'pareto', 'gamma']
                dis_dist_names = ['poisson']
                params = {}
                dist_results = []
                try:
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
                    plt.subplot(3, 4, vid)
                    plt.tight_layout()
                    plt.gca().set_title(str(dk), size=8)
                    plt.axis('off')
                    # sns.distplot(dv)
                    plt.text(0.1, 0.2, str(best_dist) + '\n' + 'best P:' + str(round(best_p, 2)) + '\n' + 'Mean:' + str(
                        round(mean_data, 2)) +
                             '\n' + 'STD:' + str(round(std_data, 2)) +
                             '\n' + 'Min:' + str(min_data) + '\n' + 'Max:' + str(
                        max_data) + '\n' + 'Coefficient of Variance:' + str(np.round(variation(dv), 2)), size=8)
                    vid += 1

                    mng = plt.get_current_fig_manager()
                    # mng.resize(*mng.window.maxsize())
                    plt.rcParams["figure.figsize"] = [20, 20]
                    plt.resize(100, 100)
                    plt.savefig('static/images/' + str(fname) + '_sdlog.csv.png', dpi=100)
                except:
                    plt.savefig('static/images/' + str(fname) + '_sdlog.csv.png', dpi=100)
        # plt.show()
        return Name_General_selected_variables



