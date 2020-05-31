<<<<<<< HEAD
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
from scipy import stats
import seaborn as sns
from scipy.stats import ks_2samp,shapiro
import networkx as nx

class Conceptual_Model:

    def calculate_overall_act(self,event_log,feature_list):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        log_duration = pd.to_timedelta(start_unit_log - end_log).seconds/3600
        unit_time_act_num_dur_dict = defaultdict(dict)
        event_log_unit = event_log
        temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Activity'])
        waiting_temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        act_num_speed_dict = defaultdict(list)
        count = 0
        output = ""

        for actname, actgroup in temp_event_log:

            if actname in feature_list:
                startforeach = []
                startdiff = []
                startforeach = (actgroup['Start Timestamp'].values)
                startforeach.sort()
                startforeach = np.array(startforeach)

                for i in range(len(startforeach)-1):
                    startdiff.append(pd.to_timedelta(startforeach[i+1]-startforeach[i]).seconds/3600)

                startdiff = np.array(startdiff)
                startdiff = startdiff
                startmax = np.max(startdiff)
                startmin = np.mean(startdiff)
                startcount = Counter(startdiff)
                frestartcount = startcount.most_common(1)

                #finished
                finishforeach = []
                finishdiff = []
                finishforeach = (actgroup['Complete Timestamp'].values)
                finishforeach.sort()
                finishforeach = np.array(finishforeach)

                for i in range(len(startforeach) - 1):
                    finishdiff.append(pd.to_timedelta(finishforeach[i + 1] - finishforeach[i]).seconds / 3600)

                finishdiff = np.array(finishdiff)
                finishdiff =finishdiff
                finishtmax = np.max(finishdiff)
                finishmin = np.mean(finishdiff)
                finishcount = Counter(finishdiff)
                frefinishcount = finishcount.most_common(1)

                list_case_per_act = []
                list_case_per_act.append(actgroup['Case ID'].nunique())

                list_complete = actgroup['Complete Timestamp'].values
                list_start = actgroup['Start Timestamp'].values
                list_act_idle_time = []
                list_act_waiting = []
                list_act_waiting = np.array(list_act_waiting)


                for i in range(len(list_complete) - 1):
                    if (list_start[i + 1] )< (list_complete[i]):
                        np.append(list_act_waiting,(abs(list_start[i + 1] - list_complete[i])))
                        list_act_idle_time.append(0)
                    else:
                        np.append(list_act_waiting,0)
                        list_act_idle_time.append(list_start[i + 1] - list_complete[i])

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_log for i in actgroup['Complete Timestamp'])
                act_inprocess_event = sum(i > end_log for i in actgroup['Complete Timestamp'])

                act_start_not_finished_event = sum(i >= end_log for i in actgroup['Complete Timestamp'])
                #0 arrival
                act_num_speed_dict[actname].append(len(act_num_event_list)/log_duration)
                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))
                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))
                #3 avg waiting time
                act_num_speed_dict[actname].append(np.mean(list_act_waiting))
                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))
                #5 waiting events
                act_num_speed_dict[actname].append(act_start_not_finished_event)
                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event/log_duration)
                #7 idle time
                act_num_speed_dict[actname].append((np.mean(list_act_idle_time)))
                #8 in process
                act_num_speed_dict[actname].append(act_inprocess_event)


                plt.subplot(3, 2, 1)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(frestartcount))
                print("On average every"+str(frestartcount)+"hour one event arrival for activity "+str(actname) )

                plt.subplot(3, 2, 2)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600))
                print("The average duration of activity "+str(actname)+": "+str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600)+" and the median is "+str(np.median(act_dur_event_list)))

                plt.subplot(3, 2, 3)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(frefinishcount))
                print("On average every" + str(frefinishcount) + "hour one event finished by activity " + str(actname))

                plt.subplot(3, 2, 4)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,np.mean(list_act_waiting))
                print("The average waiting time of activity " + str(actname) + ": " + str(np.mean(list_act_waiting)) + " and the median is " + str(np.median(list_act_waiting)))

                output = output+'\n' + "On average every"+str(frestartcount)+"hour one event arrival for activity "+str(actname) +'\n'+\
                         "The average duration of activity "+str(actname)+": "+str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600)+" and the median is "+str(np.median(act_dur_event_list))+'\n' +\
                         "On average every" + str(frefinishcount) + "hour one event finished by activity " + str(actname)+'\n'+ \
                         "The average waiting time of activity " + str(actname) + ": " + str(np.mean(list_act_waiting)) + " and the median is " + str(np.median(list_act_waiting))+'\n'
                count += 1

        """
        for caseid, casegroup in waiting_temp_event_log:
            # 8 in process
            act_num_speed_dict[actname].append()
        """
        return act_num_speed_dict,output

    def break_log_for_cases(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit

        temp_case_log = event_log.sort_values(['Start Timestamp'],ascending=True).groupby('Case ID').head(1)
        temp_case_log = temp_case_log['Start Timestamp'].values
        count_start_case_per_unit = 0
        count = 0
        list_start = []
        unit_time_res_num_dur_dict = defaultdict(dict)
        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
            case_num_dur_dict = defaultdict(list)
            num_case_per_unit = temp_event_log.ngroups

            for rcase, rcasegroup in temp_event_log:
                list_start.append(rcasegroup['Start Timestamp'].values)
                count_start_case_per_unit= ([s for s in list_start if s in temp_case_log])
            count = count +1
        case_num_dur_dict[count].update(count_start_case_per_unit)
        return case_num_dur_dict

    def break_log_for_res(self, time_unit, event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Resource'])
            case_temp_event_log = event_log_unit.groupby(['Case ID'])
            act_num_speed_dict = defaultdict(list)
            casegroup_act_dict = defaultdict(list)
            casegroup_arrival_act_dict = defaultdict(list)

            for casename, casegroup in case_temp_event_log:
                casegroup = casegroup.sort_values(['Start Timestamp'])
                casegroup_act_list = casegroup['Resource'].values
                casegroup_start_list = casegroup['Start Timestamp'].values
                casegroup_comp_list = casegroup['Complete Timestamp'].values

                for caseact in casegroup_act_list:
                    ix0 = casegroup_act_list.tolist().index(caseact)
                    ix1 = ix0 + 1

                    # if len(casegroup_act_list) == 1:
                    #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(
                            casegroup_comp_list[ix0] - casegroup_start_list[ix1]).seconds / 3600
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
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                # 4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

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

    """
    def break_log_for_res(self, time_unit, event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_res_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Complete Timestamp'],ascending=True).groupby(['Resource'])
            res_num_dur_dict = defaultdict(list)

            for rname , rgroup in temp_event_log:

                list_case_per_res = []
                list_case_per_res.append(rgroup['Case ID'].nunique())

                list_complete = rgroup['Complete Timestamp'].values
                list_start = rgroup['Start Timestamp'].values
                list_res_idle_time = []
                list_res_waiting = []
                for i in range(len(list_complete)-1):
                    if pd.to_timedelta(list_start[i+1]-list_complete[i]).seconds <0:
                        list_res_waiting.append(abs(list_start[i+1]-list_complete[i]))
                        list_res_idle_time.append(0)
                    else:
                        list_res_waiting.append(0)
                        list_res_idle_time.append(list_start[i+1]-list_complete[i])

                res_dur_event_list = []
                res_num_event_list = []
                temp_res_num_event_list = rgroup['Start Timestamp']
                res_num_event_list = temp_res_num_event_list.values
                temp_res_dur_event_list = rgroup['Event Duration']
                res_dur_event_list = temp_res_dur_event_list.values
                res_finish_event = sum(i < end_unit_log for i in rgroup['Complete Timestamp'])
                #0 arrival
                res_num_dur_dict[rname].append(len(res_num_event_list))
                #1 avg-duration
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(res_dur_event_list)))
                #2 whole_duraiton
                res_num_dur_dict[rname].append(pd.to_timedelta(np.sum(res_dur_event_list)))
                #3 avg_waiting time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(list_res_waiting)))
                #4 whole_waiting_time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.sum(list_res_waiting)))
                #5 waiting_events
                res_num_dur_dict[rname].append(len(res_num_event_list)-res_finish_event)
                #6 finished_events
                res_num_dur_dict[rname].append(res_finish_event)
                #7 idle_time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(list_res_idle_time)))
                #inprocess_events

            unit_time_res_num_dur_dict[count].update(res_num_dur_dict)
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit
            count = count + 1

        return unit_time_res_num_dur_dict
    """

    def break_log_for_act(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
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
                    ix1 = ix0+1

                    #if len(casegroup_act_list) == 1:
                     #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(casegroup_start_list[ix1] - casegroup_comp_list[ix0] ).seconds/3600
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
                if len(list_act_idle_time) == 0 :
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting)==0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                #0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1 :
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                #act_num_speed_dict[actname].append(len(act_num_event_list))

                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))

                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))

                #3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact,caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                #5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                #7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                #8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)
                # 9 number of unique resource
               # act_num_speed_dict[actname].append(actgroup['Resource'].nunique())
                # 10 number of involved resource
                #act_num_speed_dict[actname].append(len(actgroup['Resource']))


            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit

        return unit_time_act_num_dur_dict

    #todo calculate the features values on aggrigated level
    def break_uptopoint_log_for_act(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
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
                    ix1 = ix0+1

                    #if len(casegroup_act_list) == 1:
                     #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(casegroup_start_list[ix1] - casegroup_comp_list[ix0] ).seconds/3600
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
                if len(list_act_idle_time) == 0 :
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting)==0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                #0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1 :
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                #act_num_speed_dict[actname].append(len(act_num_event_list))

                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))

                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))

                #3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact,caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                #5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                #7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                #8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)

            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            end_unit_log = end_unit_log + time_unit

        return unit_time_act_num_dur_dict

    def check_time_unit(self,time_unit,dict_values):

        variables = dict_values
        count = 0
        for v in variables:
            stats, p = shapiro(v)
            if p > 0.05 :
                count +1
                print("with time unit "+str(time_unit)+"variable is normally distributed")
            else:
                print("with time unit "+str(time_unit)+"variable is not normally distributed")
        var_consistency_in_tu = 100(count/len(variables))
        return var_consistency_in_tu

    def validation (self, real_data, simulation_data):
        simulation_list = []
        simulation_list = simulation_data
        simulation_list = simulation_list[:-1]
        real_list = []
        real_list = real_data

    #data description
        plt.subplot(3,3,1)
        data = [real_list, simulation_list]
        sns.boxplot(real_list)
        plt.subplot(3, 3,2)
        sns.boxplot(simulation_list,color="orange")
        plt.subplot(3,3,4)
        sns.distplot(real_list,rug=True, hist=False)
        plt.subplot(3, 3, 5)
        sns.distplot(simulation_list, color="orange",rug=True, hist=False)
        statistical_compare = pd.DataFrame([[np.mean(real_list),np.std(real_list),np.std(real_list)/np.mean(real_list)],
                      [np.mean(simulation_list),np.std(simulation_list),np.std(simulation_list)/np.mean(simulation_list)]],index=['reality','simulated'],columns=['mean','STD','CV'])
        plt.subplot(3,3,3)
        plt.text(0.1,0.2,statistical_compare,fontsize=12)

    #Test Distribution of DAta
        sta, p_value = ks_2samp(real_list,simulation_list)
        plt.subplot(3, 3, (6))
        plt.text(0.1, 0.3, "P value of the Kolmogrov Test:\n"+str(p_value), fontsize=12)


    #Test PairWise
        diff_list = []
        diff_list = np.array(diff_list)
        diff_list = np.array(real_list) - np.array(simulation_list)
        plt.subplot(3, 3, (7,8))
        sns.distplot(diff_list,color='green', rug=True, hist=False)
        plt.subplot(3, 3, 9)
        acceptance_rate = (sum(i <= 0.20 for i in diff_list) / len(real_list)) * 100
        #diff_normality_test
        p , stat = stats.shapiro(diff_list)
        if p >0.5:
            plt.text(0.1,0.3, "differenc is a normal distribution with mean of: \n"+str(np.mean(diff_list)))
        else:
            plt.text(0.1, 0.3, "differenc is not a normal distribution with mean of: \n" + str(np.mean(diff_list)))
        #plt.show()

    #Test consistencty Realtion of independent and dependent features (Correlation Direction of movemnet)
        var1_real=[]
        var2_real=[]
        var_result_real = []


        var1_sim = []
        var2_sim = []
        var_result_sim = []

        return

    def check_activity (self, event_log):

        #create adjancy matrix of activities
        matrix = pd.DataFrame(np.zeros(shape=(event_log['Activity'].nunique(),event_log['Activity'].nunique())),
                              columns=event_log['Activity'].unique(), index=event_log['Activity'].unique())
        temp_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby('Case ID')
        trace = {}
        for case,casegroup in temp_log:
            trace.update({case: casegroup['Activity'].values })
        for key,val in trace.items():
            i = 0
            while i < (len(val)-1):
                matrix[val[i+1]][val[i]] += 1
                i += 1
        temp_list = event_log.groupby(['Activity'])['Event Duration']
        activity_duaration = temp_list.sum()
        activity_count = temp_list.size()
        activity_percentage = (pd.to_timedelta(activity_duaration.values).days)/activity_count
        most_delay_activity = activity_percentage.idxmax()

        return matrix

    def write_2file(self,variable_dict):
        data = variable_dict
        data = pd.DataFrame({"varibale_name":[],})
        writer = pd.ExcelWriter("SDLOG.xlsx")
        data.to_excel(writer)
        writer.save()
        return

    def create_model(self,f1,f2,f3):
        features_dict = defaultdict(list)
        f_list =[]
        np.array(f_list)
        feature_df = pd.DataFrame(np.column_stack([f1,f2,f3]) , columns = ['f1','f2','f3'])
        cor = pd.DataFrame(feature_df.corr())
        #sns.heatmap(feature_df.corr(), annot=True)
        #sns.pairplot(feature_df)

        #Graph of related features Networks Lib
        G = nx.Graph()
        for idx, row in cor.iterrows():
            G.add_node(idx)

        for idx in cor.index:
            for con in cor.columns:
                if abs(cor[idx][con]) > 0.98 and cor[idx][con] != 1:
                    G.add_edge(idx,con)
        plt.axis('off')
        nx.draw_networkx(G)
        #plt.show()

        return feature_df

    def find_log_act(self,event_log):
        event_log = event_log.groupby(['Activity'])['Event Duration']
        sum_duration = event_log.sum()
        coun_duraion =event_log.size()
        mean_duration = sum_duration/coun_duraion
        max_duration = max(mean_duration)
        max_dur_act =  mean_duration.idxmax()
        return max_dur_act

    def case_arrival_interval(self, event_log):
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        finish_temp_event_log = event_log.sort_values(['Complete Timestamp'], ascending=True).groupby(['Case ID'])
        startdiff = []
        startforeach = []
        for actname, actgroup in temp_event_log:
            startforeach.append(actgroup['Start Timestamp'].values[0])
        startforeach.sort()

        for i in range(len(startforeach) - 1):
            startdiff.append((startforeach[i + 1] - startforeach[i]))

        plt.plot(pd.to_timedelta(startdiff).seconds / 60)
        counter_intervals = Counter(pd.to_timedelta(startdiff).seconds // 3600)
        x = np.linspace(0,100,1000)
        plt.scatter(x, startdiff)
        #plt.show()
        return

=======
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
from scipy import stats
import seaborn as sns
from scipy.stats import ks_2samp,shapiro
import networkx as nx

class Conceptual_Model:

    def calculate_overall_act(self,event_log,feature_list):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        log_duration = pd.to_timedelta(start_unit_log - end_log).seconds/3600
        unit_time_act_num_dur_dict = defaultdict(dict)
        event_log_unit = event_log
        temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Activity'])
        waiting_temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        act_num_speed_dict = defaultdict(list)
        count = 0
        output = ""

        for actname, actgroup in temp_event_log:

            if actname in feature_list:
                startforeach = []
                startdiff = []
                startforeach = (actgroup['Start Timestamp'].values)
                startforeach.sort()
                startforeach = np.array(startforeach)

                for i in range(len(startforeach)-1):
                    startdiff.append(pd.to_timedelta(startforeach[i+1]-startforeach[i]).seconds/3600)

                startdiff = np.array(startdiff)
                startdiff = startdiff
                startmax = np.max(startdiff)
                startmin = np.mean(startdiff)
                startcount = Counter(startdiff)
                frestartcount = startcount.most_common(1)

                #finished
                finishforeach = []
                finishdiff = []
                finishforeach = (actgroup['Complete Timestamp'].values)
                finishforeach.sort()
                finishforeach = np.array(finishforeach)

                for i in range(len(startforeach) - 1):
                    finishdiff.append(pd.to_timedelta(finishforeach[i + 1] - finishforeach[i]).seconds / 3600)

                finishdiff = np.array(finishdiff)
                finishdiff =finishdiff
                finishtmax = np.max(finishdiff)
                finishmin = np.mean(finishdiff)
                finishcount = Counter(finishdiff)
                frefinishcount = finishcount.most_common(1)

                list_case_per_act = []
                list_case_per_act.append(actgroup['Case ID'].nunique())

                list_complete = actgroup['Complete Timestamp'].values
                list_start = actgroup['Start Timestamp'].values
                list_act_idle_time = []
                list_act_waiting = []
                list_act_waiting = np.array(list_act_waiting)


                for i in range(len(list_complete) - 1):
                    if (list_start[i + 1] )< (list_complete[i]):
                        np.append(list_act_waiting,(abs(list_start[i + 1] - list_complete[i])))
                        list_act_idle_time.append(0)
                    else:
                        np.append(list_act_waiting,0)
                        list_act_idle_time.append(list_start[i + 1] - list_complete[i])

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_log for i in actgroup['Complete Timestamp'])
                act_inprocess_event = sum(i > end_log for i in actgroup['Complete Timestamp'])

                act_start_not_finished_event = sum(i >= end_log for i in actgroup['Complete Timestamp'])
                #0 arrival
                act_num_speed_dict[actname].append(len(act_num_event_list)/log_duration)
                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))
                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))
                #3 avg waiting time
                act_num_speed_dict[actname].append(np.mean(list_act_waiting))
                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))
                #5 waiting events
                act_num_speed_dict[actname].append(act_start_not_finished_event)
                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event/log_duration)
                #7 idle time
                act_num_speed_dict[actname].append((np.mean(list_act_idle_time)))
                #8 in process
                act_num_speed_dict[actname].append(act_inprocess_event)


                plt.subplot(3, 2, 1)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(frestartcount))
                print("On average every"+str(frestartcount)+"hour one event arrival for activity "+str(actname) )

                plt.subplot(3, 2, 2)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600))
                print("The average duration of activity "+str(actname)+": "+str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600)+" and the median is "+str(np.median(act_dur_event_list)))

                plt.subplot(3, 2, 3)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,str(frefinishcount))
                print("On average every" + str(frefinishcount) + "hour one event finished by activity " + str(actname))

                plt.subplot(3, 2, 4)
                plt.gca().set_title(actname, size=10)
                plt.text(0.1, 0.2,np.mean(list_act_waiting))
                print("The average waiting time of activity " + str(actname) + ": " + str(np.mean(list_act_waiting)) + " and the median is " + str(np.median(list_act_waiting)))

                output = output+'\n' + "On average every"+str(frestartcount)+"hour one event arrival for activity "+str(actname) +'\n'+\
                         "The average duration of activity "+str(actname)+": "+str(pd.to_timedelta(np.mean(act_dur_event_list)).seconds/3600)+" and the median is "+str(np.median(act_dur_event_list))+'\n' +\
                         "On average every" + str(frefinishcount) + "hour one event finished by activity " + str(actname)+'\n'+ \
                         "The average waiting time of activity " + str(actname) + ": " + str(np.mean(list_act_waiting)) + " and the median is " + str(np.median(list_act_waiting))+'\n'
                count += 1

        """
        for caseid, casegroup in waiting_temp_event_log:
            # 8 in process
            act_num_speed_dict[actname].append()
        """
        return act_num_speed_dict,output

    def break_log_for_cases(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit

        temp_case_log = event_log.sort_values(['Start Timestamp'],ascending=True).groupby('Case ID').head(1)
        temp_case_log = temp_case_log['Start Timestamp'].values
        count_start_case_per_unit = 0
        count = 0
        list_start = []
        unit_time_res_num_dur_dict = defaultdict(dict)
        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
            case_num_dur_dict = defaultdict(list)
            num_case_per_unit = temp_event_log.ngroups

            for rcase, rcasegroup in temp_event_log:
                list_start.append(rcasegroup['Start Timestamp'].values)
                count_start_case_per_unit= ([s for s in list_start if s in temp_case_log])
            count = count +1
        case_num_dur_dict[count].update(count_start_case_per_unit)
        return case_num_dur_dict

    def break_log_for_res(self, time_unit, event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Start Timestamp'], ascending=True).groupby(['Resource'])
            case_temp_event_log = event_log_unit.groupby(['Case ID'])
            act_num_speed_dict = defaultdict(list)
            casegroup_act_dict = defaultdict(list)
            casegroup_arrival_act_dict = defaultdict(list)

            for casename, casegroup in case_temp_event_log:
                casegroup = casegroup.sort_values(['Start Timestamp'])
                casegroup_act_list = casegroup['Resource'].values
                casegroup_start_list = casegroup['Start Timestamp'].values
                casegroup_comp_list = casegroup['Complete Timestamp'].values

                for caseact in casegroup_act_list:
                    ix0 = casegroup_act_list.tolist().index(caseact)
                    ix1 = ix0 + 1

                    # if len(casegroup_act_list) == 1:
                    #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(
                            casegroup_comp_list[ix0] - casegroup_start_list[ix1]).seconds / 3600
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
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                # 4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

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

    """
    def break_log_for_res(self, time_unit, event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_res_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
            temp_event_log = event_log_unit.sort_values(['Complete Timestamp'],ascending=True).groupby(['Resource'])
            res_num_dur_dict = defaultdict(list)

            for rname , rgroup in temp_event_log:

                list_case_per_res = []
                list_case_per_res.append(rgroup['Case ID'].nunique())

                list_complete = rgroup['Complete Timestamp'].values
                list_start = rgroup['Start Timestamp'].values
                list_res_idle_time = []
                list_res_waiting = []
                for i in range(len(list_complete)-1):
                    if pd.to_timedelta(list_start[i+1]-list_complete[i]).seconds <0:
                        list_res_waiting.append(abs(list_start[i+1]-list_complete[i]))
                        list_res_idle_time.append(0)
                    else:
                        list_res_waiting.append(0)
                        list_res_idle_time.append(list_start[i+1]-list_complete[i])

                res_dur_event_list = []
                res_num_event_list = []
                temp_res_num_event_list = rgroup['Start Timestamp']
                res_num_event_list = temp_res_num_event_list.values
                temp_res_dur_event_list = rgroup['Event Duration']
                res_dur_event_list = temp_res_dur_event_list.values
                res_finish_event = sum(i < end_unit_log for i in rgroup['Complete Timestamp'])
                #0 arrival
                res_num_dur_dict[rname].append(len(res_num_event_list))
                #1 avg-duration
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(res_dur_event_list)))
                #2 whole_duraiton
                res_num_dur_dict[rname].append(pd.to_timedelta(np.sum(res_dur_event_list)))
                #3 avg_waiting time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(list_res_waiting)))
                #4 whole_waiting_time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.sum(list_res_waiting)))
                #5 waiting_events
                res_num_dur_dict[rname].append(len(res_num_event_list)-res_finish_event)
                #6 finished_events
                res_num_dur_dict[rname].append(res_finish_event)
                #7 idle_time
                res_num_dur_dict[rname].append(pd.to_timedelta(np.mean(list_res_idle_time)))
                #inprocess_events

            unit_time_res_num_dur_dict[count].update(res_num_dur_dict)
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit
            count = count + 1

        return unit_time_res_num_dur_dict
    """

    def break_log_for_act(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
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
                    ix1 = ix0+1

                    #if len(casegroup_act_list) == 1:
                     #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(casegroup_start_list[ix1] - casegroup_comp_list[ix0] ).seconds/3600
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
                if len(list_act_idle_time) == 0 :
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting)==0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                #0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1 :
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                #act_num_speed_dict[actname].append(len(act_num_event_list))

                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))

                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))

                #3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact,caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                #5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                #7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                #8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)
                # 9 number of unique resource
               # act_num_speed_dict[actname].append(actgroup['Resource'].nunique())
                # 10 number of involved resource
                #act_num_speed_dict[actname].append(len(actgroup['Resource']))


            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            start_unit_log = end_unit_log
            end_unit_log = start_unit_log + time_unit

        return unit_time_act_num_dur_dict

    #todo calculate the features values on aggrigated level
    def break_uptopoint_log_for_act(self,time_unit,event_log):

        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        end_unit_log = start_unit_log + time_unit
        count = 0
        unit_time_act_num_dur_dict = defaultdict(dict)

        while end_unit_log <= (end_log - time_unit):
            event_log_unit = event_log[event_log['Start Timestamp'].between(start_unit_log, end_unit_log)]
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
                    ix1 = ix0+1

                    #if len(casegroup_act_list) == 1:
                     #   casegroup_arrival_act_dict[casegroup_act_list[ix1]].append(1)

                    if ix1 < len(casegroup_act_list):
                        case_act_diff = pd.to_timedelta(casegroup_start_list[ix1] - casegroup_comp_list[ix0] ).seconds/3600
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
                if len(list_act_idle_time) == 0 :
                    list_act_idle_time.append(pd.to_timedelta(0))
                if len(list_act_waiting)==0:
                    list_act_waiting.append(pd.to_timedelta(0))

                act_dur_event_list = []
                act_num_event_list = []
                temp_act_num_event_list = actgroup['Start Timestamp']
                act_num_event_list = temp_act_num_event_list.values
                temp_act_dur_event_list = actgroup['Event Duration']
                act_dur_event_list = temp_act_dur_event_list.values
                act_finish_event = sum(i < end_unit_log for i in actgroup['Complete Timestamp'])
                act_start_not_finished_event = sum(j >= end_unit_log for j in actgroup['Complete Timestamp'])

                #0 arrival
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 1 :
                    act_num_speed_dict[actname].append(len(act_num_event_list))
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(len((casegroup_act_dict[caseidact])))
                #act_num_speed_dict[actname].append(len(act_num_event_list))

                #1 avg duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(act_dur_event_list)))

                #2 whole duration
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(act_dur_event_list)))

                #3 avg waiting time
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 4:
                    act_num_speed_dict[actname].append((0))
                else:
                    for caseidact,caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append((np.mean(casegroup_act_dict[caseidact])))

                #4 whole waiting time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.sum(list_act_waiting)))

                #5 waiting events
                if actname not in casegroup_act_dict and len(act_num_speed_dict[actname]) < 6:
                    act_num_speed_dict[actname].append(0)
                else:
                    for caseidact, caseidacgr in casegroup_act_dict.items():
                        if actname == caseidact:
                            act_num_speed_dict[actname].append(np.count_nonzero(casegroup_act_dict[caseidact]))

                #6 finished events
                act_num_speed_dict[actname].append(act_finish_event)
                #7 idle time
                act_num_speed_dict[actname].append(pd.to_timedelta(np.mean(list_act_idle_time)))
                #8 events in process
                act_num_speed_dict[actname].append(act_start_not_finished_event)

            unit_time_act_num_dur_dict[count].update(act_num_speed_dict)
            count = count + 1
            end_unit_log = end_unit_log + time_unit

        return unit_time_act_num_dur_dict

    def check_time_unit(self,time_unit,dict_values):

        variables = dict_values
        count = 0
        for v in variables:
            stats, p = shapiro(v)
            if p > 0.05 :
                count +1
                print("with time unit "+str(time_unit)+"variable is normally distributed")
            else:
                print("with time unit "+str(time_unit)+"variable is not normally distributed")
        var_consistency_in_tu = 100(count/len(variables))
        return var_consistency_in_tu

    def validation (self, real_data, simulation_data):
        simulation_list = []
        simulation_list = simulation_data
        simulation_list = simulation_list[:-1]
        real_list = []
        real_list = real_data

    #data description
        plt.subplot(3,3,1)
        data = [real_list, simulation_list]
        sns.boxplot(real_list)
        plt.subplot(3, 3,2)
        sns.boxplot(simulation_list,color="orange")
        plt.subplot(3,3,4)
        sns.distplot(real_list,rug=True, hist=False)
        plt.subplot(3, 3, 5)
        sns.distplot(simulation_list, color="orange",rug=True, hist=False)
        statistical_compare = pd.DataFrame([[np.mean(real_list),np.std(real_list),np.std(real_list)/np.mean(real_list)],
                      [np.mean(simulation_list),np.std(simulation_list),np.std(simulation_list)/np.mean(simulation_list)]],index=['reality','simulated'],columns=['mean','STD','CV'])
        plt.subplot(3,3,3)
        plt.text(0.1,0.2,statistical_compare,fontsize=12)

    #Test Distribution of DAta
        sta, p_value = ks_2samp(real_list,simulation_list)
        plt.subplot(3, 3, (6))
        plt.text(0.1, 0.3, "P value of the Kolmogrov Test:\n"+str(p_value), fontsize=12)


    #Test PairWise
        diff_list = []
        diff_list = np.array(diff_list)
        diff_list = np.array(real_list) - np.array(simulation_list)
        plt.subplot(3, 3, (7,8))
        sns.distplot(diff_list,color='green', rug=True, hist=False)
        plt.subplot(3, 3, 9)
        acceptance_rate = (sum(i <= 0.20 for i in diff_list) / len(real_list)) * 100
        #diff_normality_test
        p , stat = stats.shapiro(diff_list)
        if p >0.5:
            plt.text(0.1,0.3, "differenc is a normal distribution with mean of: \n"+str(np.mean(diff_list)))
        else:
            plt.text(0.1, 0.3, "differenc is not a normal distribution with mean of: \n" + str(np.mean(diff_list)))
        #plt.show()

    #Test consistencty Realtion of independent and dependent features (Correlation Direction of movemnet)
        var1_real=[]
        var2_real=[]
        var_result_real = []


        var1_sim = []
        var2_sim = []
        var_result_sim = []

        return

    def check_activity (self, event_log):

        #create adjancy matrix of activities
        matrix = pd.DataFrame(np.zeros(shape=(event_log['Activity'].nunique(),event_log['Activity'].nunique())),
                              columns=event_log['Activity'].unique(), index=event_log['Activity'].unique())
        temp_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby('Case ID')
        trace = {}
        for case,casegroup in temp_log:
            trace.update({case: casegroup['Activity'].values })
        for key,val in trace.items():
            i = 0
            while i < (len(val)-1):
                matrix[val[i+1]][val[i]] += 1
                i += 1
        temp_list = event_log.groupby(['Activity'])['Event Duration']
        activity_duaration = temp_list.sum()
        activity_count = temp_list.size()
        activity_percentage = (pd.to_timedelta(activity_duaration.values).days)/activity_count
        most_delay_activity = activity_percentage.idxmax()

        return matrix

    def write_2file(self,variable_dict):
        data = variable_dict
        data = pd.DataFrame({"varibale_name":[],})
        writer = pd.ExcelWriter("SDLOG.xlsx")
        data.to_excel(writer)
        writer.save()
        return

    def create_model(self,f1,f2,f3):
        features_dict = defaultdict(list)
        f_list =[]
        np.array(f_list)
        feature_df = pd.DataFrame(np.column_stack([f1,f2,f3]) , columns = ['f1','f2','f3'])
        cor = pd.DataFrame(feature_df.corr())
        #sns.heatmap(feature_df.corr(), annot=True)
        #sns.pairplot(feature_df)

        #Graph of related features Networks Lib
        G = nx.Graph()
        for idx, row in cor.iterrows():
            G.add_node(idx)

        for idx in cor.index:
            for con in cor.columns:
                if abs(cor[idx][con]) > 0.98 and cor[idx][con] != 1:
                    G.add_edge(idx,con)
        plt.axis('off')
        nx.draw_networkx(G)
        #plt.show()

        return feature_df

    def find_log_act(self,event_log):
        event_log = event_log.groupby(['Activity'])['Event Duration']
        sum_duration = event_log.sum()
        coun_duraion =event_log.size()
        mean_duration = sum_duration/coun_duraion
        max_duration = max(mean_duration)
        max_dur_act =  mean_duration.idxmax()
        return max_dur_act

    def case_arrival_interval(self, event_log):
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        finish_temp_event_log = event_log.sort_values(['Complete Timestamp'], ascending=True).groupby(['Case ID'])
        startdiff = []
        startforeach = []
        for actname, actgroup in temp_event_log:
            startforeach.append(actgroup['Start Timestamp'].values[0])
        startforeach.sort()

        for i in range(len(startforeach) - 1):
            startdiff.append((startforeach[i + 1] - startforeach[i]))

        plt.plot(pd.to_timedelta(startdiff).seconds / 60)
        counter_intervals = Counter(pd.to_timedelta(startdiff).seconds // 3600)
        x = np.linspace(0,100,1000)
        plt.scatter(x, startdiff)
        #plt.show()
        return

>>>>>>> ba9fe68d53340f50e0572b489d2912038af8c351
