import pandas as pd
import numpy as np
import os
from collections import Counter,defaultdict
import scipy.stats
from pyvis.network import Network


class organization_aspect:

    def find_resource(self, event_log):
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration

        freq_act_res_matrix = pd.DataFrame(
            np.zeros(shape=(len(event_log['Resource'].unique()), len(event_log['Activity'].unique()))),
            columns=event_log['Activity'].unique(), index=event_log['Resource'].unique())
        dur_act_res_matrix = pd.DataFrame(
            np.zeros(shape=(len(event_log['Resource'].unique()), len(event_log['Activity'].unique()))),
            columns=event_log['Activity'].unique(), index=event_log['Resource'].unique())

        act_groupy = event_log.groupby('Activity')
        for name, group in act_groupy:
            resgroup = group.groupby('Resource')['Event Duration']
            res_per_act_freq = resgroup.size()
            res_per_act_sum = resgroup.sum()


            for res in res_per_act_freq.keys():
                freq_act_res_matrix[name][res] = res_per_act_freq.get(res)
                if res_per_act_freq.get(res) != 0 and res_per_act_sum.get(res) != 0:
                    dur_act_res_matrix[name][res] = pd.to_timedelta(
                        (res_per_act_sum.get(res)) / res_per_act_freq.get(res)).seconds / 3600

        return freq_act_res_matrix

    def find_roles(self, freq_act_res_matrix):
        resource_index_dict = {}
        act_list = freq_act_res_matrix.columns
        # todo list of resources in one org as Key, value Acts in the org
        org_act_res_dict = defaultdict(list)
        for a in act_list:
            org_act_res_dict[tuple(freq_act_res_matrix.index[freq_act_res_matrix[a] != 0].tolist())].append(a)

        rows = len(freq_act_res_matrix.index)
        for row in range((rows)):
            resource_index_dict[row] = freq_act_res_matrix.index[row]
        res_res_similarity = pd.DataFrame(np.zeros(shape=(rows, rows)), columns=freq_act_res_matrix.index,
                                          index=freq_act_res_matrix.index)

        i = 0
        while i < rows:
            j = 0
            while j < rows:
                x = freq_act_res_matrix.iloc[[i]].values
                x = x[0]
                y = freq_act_res_matrix.iloc[[j]].values
                y = y[0]
                p_core, p_value = scipy.stats.pearsonr(x, y)
                res_res_similarity[resource_index_dict.get(i)][resource_index_dict.get(j)] = p_core
                j += 1
            i += 1

        return res_res_similarity, org_act_res_dict

    def filter_log_org(self, event_log, Org):
        filtered_event_log = event_log.where(event_log['Resource'].isin(Org))
        filtered_event_log = filtered_event_log.dropna()
        return filtered_event_log

    def filter_log_act(self, event_log, act):
        filtered_event_log = event_log.where(event_log['Activity'].isin(act))
        filtered_event_log = filtered_event_log.dropna()
        return filtered_event_log

    def create_matrix(self, event_log):
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration

        temp_act_log = event_log.groupby(['Activity'])

        # create adjancy matrix of activities
        matrix = pd.DataFrame(np.zeros(shape=(event_log['Activity'].nunique(), event_log['Activity'].nunique())),
                              columns=event_log['Activity'].unique(), index=event_log['Activity'].unique())
        temp_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby('Case ID')
        trace = {}

        for case, casegroup in temp_log:
            trace.update({case: casegroup['Activity'].values})

        for key, val in trace.items():
            i = 0
            while i < (len(val) - 1):
                matrix[val[i + 1]][val[i]] += 1
                i += 1
        for inm in matrix.columns:
            matrix[inm] = matrix[inm]/sum(matrix[inm].values)
        return matrix

    def create_DFG(self, matrix):
        cwd = os.getcwd()
        for col in matrix.columns:
            matrix.loc[matrix[col] < 0.7, col] = 0

        G = Network( height="800px",
                 width="800px",directed=True)
        matrixt = matrix.T
        for act in matrix.columns:
            G.add_node(act, shape='box', label=str(act))
        G.add_node("start", shape='box', label="start", color='green')
        G.add_node("end", shape='box', label="end", color="red")
        for act in matrix.columns:
            temp_in = matrix[act]
            for intemp in temp_in.iteritems():
                if intemp[1] != 0:
                    #G.add_node(intemp[0], shape='box', label=str(intemp[0]))
                    G.add_edge(intemp[0], act)
        for acts in matrixt.columns:
            temp_in = matrix[acts]
            sum_temp_in_values = np.sum(temp_in.values)
            if sum_temp_in_values == 0:
                G.add_edge("start", acts)

        for acte in matrixt.columns:
            temp_in = matrixt[acte]
            sum_temp_in_values = np.sum(temp_in.values)
            if sum_temp_in_values == 0:
                G.add_edge(acte, "end")
        #G.save_graph("DFG.html")
        path=os.path.join('templates', 'mygraph.html')
        #G.save_graph(str(cwd)+"\\templates\mygraph.html")
        G.save_graph(path)

        return

    def create_pmodel_org_level(self,event_log):
        matrix=self.create_matrix(event_log)
        freq_act_res_matrix=self.find_resource(event_log)
        _,org_dict=self.find_roles(freq_act_res_matrix)
        new_matrix=matrix.copy()
        for act in matrix.columns:
            for k, val in org_dict.items():
                if act in val:
                    new_matrix = new_matrix.rename(index={act: ",".join(val)}, columns={act: ",".join(val)})

        new_matrix=new_matrix.groupby(level=0, axis=1).sum()
        new_matrix=new_matrix.groupby(level=0, axis=0).sum()
        self.create_DFG(new_matrix)

        return

