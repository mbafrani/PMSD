import csv
import pysd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
listact = []

#file_dir = input(r'address:')
df = pd.read_csv(r"C:\Users\mahsa\Desktop\From Case\PROMCSCar.csv")

for index, row in df.iterrows():
    id = row["event"].split('{', 1)[1].split('}')[0]
    #row["case"] = id.split('=',1)[1]
    if id.split(',')[0].split('=')[1] ==' []':
        row['case'] = id.split(',')[1].split('=')[1].split(',')[0]
    else:
        row["case"]= id.split(',')[0].split('=')[1]
    row["event"] = row["event"].split('.', 1)[1].split('{')[0]
    df.iloc[index] = row
df.to_csv('PrOMCSCarClean.csv', encoding='utf-8', index=False)

dffile = pd.read_csv('PrOMCSCarClean.csv')

dfsorted = dffile.sort_values(["Case ID","startTime" ])
dfsorted =dfsorted.reset_index(drop=True)
df_copy = pd.DataFrame(columns=dfsorted.columns)
i = 0
while i<30000:
    row1 = dfsorted.iloc[[i]]
    row2 = dfsorted.iloc[[i+1]]
    case1 = row1["case"].values
    name1 = row1["event"].values
    name2 = row2["event"].values
    if row1["event"].values != "Check_ticket.A1" and   row1["event"].values != "Check_ticket.A12" and  row1["event"].values !="Decide.A2" and  row1["event"].values !="Decide.A2e":
        df_copy = df_copy.append(row1, ignore_index=True)
    if row1["case"].values == row2["case"].values and name1 == "Check_ticket.A1" and name2 == "Check_ticket.A12":
        row1["completeTime"] = row2["completeTime"].values
        df_copy =df_copy.append(row1,ignore_index=True)
    if row1["case"].values == row2["case"].values and name1 == "Decide.A2" and name2 == "Decide.A2e":
        row1["completeTime"] = row2["completeTime"].values
        df_copy = df_copy.append(row1, ignore_index=True)
    i = i+1

df_copy.to_csv('8-15-18-2.csv', encoding='utf-8', index=False)

"""

dfcase = df.groupby(['Case ID'])
for name,group in dfcase:
    for g in group['Activity']:
        if g =="Secondvisit":
            df['Complete Timestamp'] = df['Activity'][]
for ac in df['Activity'].values:
    if str(ac).split('_')[0] == 'W':
        listact.append(ac)

df_new = df.loc[df['Activity'].isin(listact)]
df_new.to_csv('new2012.csv', encoding='utf-8', index=False)
"""
