import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from statsmodels.tsa.arima_model import ARIMA
import csv
import re
import os

class TW_Analysis:

    #Only calculate for the list of given tws
    def TW_discovery_process_calculation_twlist(self, event_log,tw_lists,aspect):
        Overall_dict = {}
        Arrival_rate_dict = {}
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'], errors='coerce')
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'], errors='coerce')
        event_log['Activity Duration'] = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        arr_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        fin_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        case_dur_temp_log = event_log.groupby(['Case ID'])
        Name_General_selected_variables_dict=[]
        for tw_list in tw_lists:
            # todo Arrival Rate of Cases per Day, Week, Month
            y = int((re.findall(r'\d+', tw_list))[0])
            z = tw_list[-1]

            if z == "H" and (24 % y == 0) or z!= 'H':

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

            # case_duration_dict = {k: v for k, v in case_duration_dict.items() if v>0.5}
            #case_duration_dict = {k: v for k, v in case_duration_dict.items() if np.abs(v - np.mean(list(case_duration_dict.values())))<= (np.std(list(case_duration_dict.values())))}

            # case_real_duration_dict = {k: v for k, v in case_real_duration_dict.items() if v > 0.5}
            # case_real_duration_dict = {k: v for k, v in case_real_duration_dict.items() if np.abs(v - np.mean(list(case_real_duration_dict.values()))) <=(np.std(list(case_real_duration_dict.values())))}

            case_duration_list = list(case_duration_dict.values())
            case_real_duration_list = list(case_real_duration_dict.values())

            case_duration_df = pd.DataFrame(case_duration_dict.items(), columns=['Start Timestamp', 'Case Duration'])
            case_duration_df = case_duration_df.sort_values('Start Timestamp')
            case_duration_df.set_index('Start Timestamp', inplace=True)
            case_duration_H_df = case_duration_df.resample(str(tw_list)).sum()
            case_duration_H_df.insert(0, 'Avg Case Duration', case_duration_df.resample(str(tw_list)).mean())
            case_duration_H_df=case_duration_H_df.fillna(0)
            case_real_duration_df = pd.DataFrame(case_real_duration_dict.items(),
                                                 columns=['Start Timestamp', 'Case Duration'])
            case_real_duration_df = case_real_duration_df.sort_values('Start Timestamp')
            case_real_duration_df.set_index('Start Timestamp', inplace=True)
            case_real_duration_H_df = case_real_duration_df.resample(str(tw_list)).sum()
            case_real_duration_H_df.insert(0, 'Avg Case Duration', case_real_duration_df.resample(str(tw_list)).mean())
            case_real_duration_H_df= case_real_duration_H_df.fillna(0)

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



            """
            finsihed_week =[]
            for sk in sklist:
                finish_counter = 0
                tem_com = temp_group_w.get(sk)
                for i in tem_com['Complete Timestamp']:
                    if sk>=i:
                        finish_counter += 1
                finsihed_week.append(finish_counter)
                """

            # TODO Number of Resources and unique resources per case
            uniqe_resource_list_per_case = []
            resource_list_per_case = []
            for rc, rgroup in case_dur_temp_log:
                resource_per_case = rgroup['Resource']
                unique_resource_per_case = np.unique(resource_per_case)
                uniqe_resource_list_per_case.append(len(unique_resource_per_case))
                resource_list_per_case.append(len(resource_per_case))

            # TODO Create Overall Dict
            average_waiting= pd.array(case_duration_H_df['Avg Case Duration'].tolist()) - pd.array(
                case_real_duration_H_df['Avg Case Duration'].tolist())
            average_waiting = np.array(average_waiting).tolist()
            average_waiting = [0 if x < 0 else x for x in average_waiting]


            Arrival_rate_dict[str(tw_list)]=Hourly['hourly'].values.tolist()


            Overall_dict["Arrival rate"] = Arrival_rate_dict
            Name_General_selected_variables_dict.append(str(aspect)+"_"+str(tw_list)+"_sdlog.csv")
            General_selected_variables_dict = {"Arrival rate"+str(tw_list): Hourly['hourly'].values.tolist(),
                                               "Finish rate"+str(tw_list): (eHourly['ehourly'].values).tolist(),
                                               "Num of unique resource"+str(tw_list): num_unique_resource_h.tolist(),
                                               "Process active time"+str(tw_list): case_duration_H_df[
                                                   'Case Duration'].tolist(),
                                               "Service time per case"+str(tw_list): case_real_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Time in process per case"+str(tw_list): case_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Waiting time in process per case"+str(tw_list):average_waiting ,
                                               "Num in process case"+str(tw_list): temp_list_inproc,
                                              }
            outputpath= os.path.join("Outputs",str(aspect)+"_"+str(tw_list)+"_sdlog.csv")
            with open(outputpath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(General_selected_variables_dict.keys())
                x = zip(*General_selected_variables_dict.values())
                xx = zip(*x)
                xxx = zip(*xx)
                writer.writerows(xxx)

        return Overall_dict,Name_General_selected_variables_dict

    def Detect_pattern_tw(self, Overall_dict, event_log,activeness):
        # TODO  Find suitable  bin  for time
        # duration of log in H, D,W,M:
        """
        end_log = pd.to_datetime(np.max(event_log["Complete Timestamp"].values))
        start_log = pd.to_datetime(np.min(event_log["Start Timestamp"].values))
        duration_log = end_log - start_log
        duration_log_H = duration_log.total_seconds() / 3600
        duration_log_D = duration_log.days
        duration_log_W = (duration_log.total_seconds() / 86400) // 7
        duration_log_M = ((duration_log.total_seconds() / 86400) // 7) // 4
        # number of instance (case, events,...):
        instance_num = event_log["Case ID"].nunique()
        # plot each differnt bin:
        bin_H = instance_num / duration_log_H
        bin_D = instance_num / duration_log_D
        bin_W = instance_num / duration_log_W
        # bin_M = instance_num/duration_log_M
        # plt.hist()
        """
        # Todo Read features from SD: dict

        TW_Dete_dict = defaultdict(list)
        number_figure = round(math.sqrt(len(Overall_dict["Arrival rate"].keys())))

        vid = 1
        plt.figure()
        tw_number =len(list(Overall_dict["Arrival rate"].keys()))
        tw_img_num= np.round(math.sqrt(tw_number),0)
        for ktw, vtw in Overall_dict["Arrival rate"].items():
            #plt.rcParams["figure.figsize"] = [8, 8]
            plt.subplot(2, 2, vid)
            plt.tight_layout()
            plt.gca().set_title("Arrival Rate Per " + str(ktw), size=8)

            from sklearn import preprocessing
            vtwn = (vtw - np.min(vtw)) / (np.max(vtw) - np.min(vtw))
            plt.plot([i for i in range(0, len(vtwn))], vtwn, label=str(ktw))
            vid = vid + 1

        if activeness =='all':
            outputpath=os.path.join("static","images","UserPattern.png")
            plt.savefig(outputpath, dpi=100)
            #plt.bar([i for i in range(0, len(vtw))], vtw)
            # plt.show()

            # TODO Detrend by removing difference:
           # diff = list()
            #for i in range(1, len(vtw)):
             #   value = vtw[i] - vtw[i - 1]
              #  diff.append(value)
            # plt.plot(diff)
            # plt.show()

            # ،TODO Find lag with maximum Corr and best TW:
        for ktw, vtw in Overall_dict["Arrival rate"].items():
            max_lag = (len(vtw)//10)
            if max_lag >0:

                plt.figure()
                temp_acorr = plt.acorr(np.array(vtw).astype(float), maxlags=max_lag)
                temp_acorr_1 = plt.acorr(np.array(vtw).astype(float), maxlags=max_lag)
                # temp_acorr = plt.acorr(np.array(vtw).astype(float))
                temp_acorr[1][temp_acorr[1] == 1.0] = 0
                TW_Dete_dict[ktw].append(np.max(temp_acorr[1]))
                index_max = np.argmax(temp_acorr[1])
                TW_Dete_dict[ktw].append(temp_acorr[0][index_max])
                TW_Dete_dict[ktw].append(temp_acorr[1][np.where(temp_acorr[0]==1)])
            else:

                TW_Dete_dict[ktw].append(0)
                TW_Dete_dict[ktw].append(0)
                TW_Dete_dict[ktw].append(0)
            # TODO find lag using CAF and PCAF
            # plot_pacf(diff, lags=7)
            # plot_acf(diff, lags=7)


        return TW_Dete_dict

    def Detect_best_user_tw(self, TW_Dete_dict, Overall_dict):

        error_delta_dict = {}
        d = {}
        for k, v in Overall_dict.get("Arrival rate").items():
            v = list(filter(lambda a: a != 0, v))
            best_lag = abs(TW_Dete_dict.get(k)[1])
            diff = list()
            for i in range(1, len(v)):
                value = v[i] - v[i - 1]
                diff.append(value)

            diff = [abs(x) + 1 for x in diff]
            if len(diff) > 5 :
                if 0<best_lag and best_lag <len(diff)//5 and best_lag<5:
                    try:
                        arima_model = ARIMA(diff, order=(best_lag, 1, 1))
                        model_fit = arima_model.fit(transparams=False)
                        t = model_fit.predict()
                        error = abs(np.mean(abs(diff[:-1] - t) / diff[:-1]))
                        error_delta_dict[str(k)] = error
                    except:
                        arima_model = ARIMA(diff, order=(best_lag, 1, 0))
                        model_fit = arima_model.fit(transparams=False)
                        t = model_fit.predict()
                        error = abs(np.mean(abs(diff[:-1] - t) / diff[:-1]))
                        error_delta_dict[str(k)] = error


                else:

                    arima_model = ARIMA(diff, order=(1, 0, 1))
                    try:
                        model_fit = arima_model.fit(start_ar_lags=2)
                        t = model_fit.predict()
                        error = abs(np.mean(abs(diff - t) / diff))
                        error_delta_dict[str(k)] = error
                    except:
                        error_delta_dict[str(k)] = -0.1
            else:
                error_delta_dict[str(k)] = -0.1

        lists = sorted(error_delta_dict.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)
        plt.figure()
        plt.plot(x, y, '--bo')
        plt.title("Error of Different Time Delta")
        plt.xlabel("Time Delta")
        plt.ylabel("PAME")
        outputpath = os.path.join("static", "images", "UserTWError.png")
        plt.savefig(outputpath, dpi=100)
        return

    def Post_process_tw(self,SD_Log, TW_Dete_dict):

        #SD_Log = pd.read_csv("General2H_sdlog.csv")
        SD_Log = SD_Log.fillna(0)
        c = str(SD_Log.columns[0])[-2:]
        temp_pattern = abs(TW_Dete_dict.get(c)[1])
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
        plt.hist(diff)
        Active_SD_Log.to_csv(os.path.join("Outputs","Active" + "_" + str(c) + "_sdlog.csv"),index=False)
        return Active_SD_Log