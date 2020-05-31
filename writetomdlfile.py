<<<<<<< HEAD
import pandas as pd
from shutil import copyfile
import numpy as np
import os
from pyvis.network import Network
import matplotlib as plt
from collections import defaultdict
import os
class creat_CFD:

    def write_cfd(self,corr_df):
        cwd = os.getcwd()
        map_dict={}
        corr_df_dict = corr_df.to_dict('dict')

        tempvarnames = ['a','b','c','d','e','f','g','h','i','j']
        copyfile(os.path.join('ModelsFormat', 'testDFD.mdl'),os.path.join('ModelsFormat', "newtestDFD.mdl"))
        f = open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r')
        filedata = f.read()
        f.close()
        i = 0
        for kv in corr_df.index.values.tolist():
            filedata = filedata.replace(','+str(tempvarnames[i]+','), ','+str(kv)+',')
            f = open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'w')
            f.write(filedata)
            f.close()
            i +=1

        with open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                tempstr = ''
                for kmm, vmm in vm.items():
                    if vmm != 0 and tempstr == '':
                        tempstr = tempstr+str(kmm)
                    elif vmm != 0 and tempstr != '':
                        tempstr = tempstr + ',' + str(kmm)

                cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')


        with open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r+') as f:
            content = f.read()
            tempplace= content.index(':L<%^E!@')
            f.seek(tempplace)
            f.write("")

        print ('done')

        #TODO: visualize CFD python

        corr_df[corr_df == 0] = 0
        G = Network(height="800px",
                 width="800px",directed=True)
        corr_dft = corr_df.T

        for act in corr_df.columns:
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
        G.save_graph(os.path.join('templates', 'mygraph.html'))
        #G.save_graph(str(os.getcwd())+"\\templates\mygraph.html")
        #G.save_graph(r"C:\Users\mahsa\PycharmProjects\SharedProjectForReviewCoopis2019\templates\cfdgraph.html")

        #TODO write the relation function into text for further use
        with open(os.path.join('ModelsFormat', "relationinCFD.txt"), 'w+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                tempstr = ''
                for kmm, vmm in vm.items():
                    if vmm != 0 and tempstr == '':
                        tempstr = tempstr+str(kmm)
                    elif vmm != 0 and tempstr != '':
                        tempstr = tempstr + ',' + str(kmm)

                cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')


        return

    def read_cdf_2sfd_stockbased(self):
        cwd = os.getcwd()
        param_ele_dict = defaultdict(list)
        sd_df = pd.read_csv(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"))
        sd_df.columns = sd_df.columns.str.replace(' ', '')
        if 10>len(sd_df.columns):
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)

                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
        if 16>len(sd_df.columns) >10:
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)
                    param_ele_dict['stock2'].append(col)
                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
                    param_ele_dict['inflow2'].append(col)
                    param_ele_dict['outflow2'].append(col)
            param_ele_dict['stock2'].append("none")
            param_ele_dict['inflow2'].append("none")
            param_ele_dict['outflow2'].append("none")

        if len(sd_df.columns) >16:
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)
                    param_ele_dict['stock2'].append(col)
                    param_ele_dict['stock3'].append(col)
                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
                    param_ele_dict['inflow2'].append(col)
                    param_ele_dict['outflow2'].append(col)
                    param_ele_dict['inflow3'].append(col)
                    param_ele_dict['outflow3'].append(col)
            param_ele_dict['inflow2'].append("none")
            param_ele_dict['outflow2'].append("none")
            param_ele_dict['stock2'].append("none")
            param_ele_dict['inflow3'].append("none")
            param_ele_dict['outflow3'].append("none")
            param_ele_dict['stock3'].append("none")

        return param_ele_dict

    def write_sfd_stockbased(self, map_dict):
        cwd = os.getcwd()
        if "stock2" in map_dict.keys():
            mainSFDfile = r"ModelsFormat\testDFD.mdl"
            newSFDfile = r"ModelsFormat\newtestDFD"

        else:
            mainSFDfile = r"ModelsFormat\1StockSFD.mdl"
            newSFDfile = r"ModelsFormat\new1StockSFD.mdl"


            tempvarnames = ['stock1', 'stock2', 'inflow1', 'outflow1', 'variable1', 'variable2', 'variable3',
                            'variable4', 'variable5', 'variable6']

            varlist = []
            with open(r"ModelsFormat\relationinCFD.txt") as fp:
                line = fp.readlines()
                for i in line:
                    if "=" in i:
                        varlist.append(i.split("=")[0])

            copyfile(mainSFDfile, newSFDfile)
            f = open(newSFDfile, 'r')
            filedata = f.read()
            f.close()
            i = 0
            for k, v in map_dict.items():
                if v in varlist:
                    varlist.pop(varlist.index(v))
                if v != 'submit':
                    filedata = filedata.replace(',' + str(k) + ',', ',' + str(v) + ',')
                    f = open(newSFDfile, 'w')
                    f.write(filedata)
                    f.close()
                i += 1

            for var in range(len(varlist)):
                filedata = filedata.replace(',' + "variable"+str(var+1) + ',', ',' + str(varlist[var]) + ',')
                f = open(newSFDfile, 'w')
                f.write(filedata)
                f.close()


            with open(r"ModelsFormat\relationinCFD.txt") as infile, open(newSFDfile, 'r+') as outfile:
                outfile.seek(0)
                for li in infile:
                    outfile.write(li)
                infile.close()
                outfile.close()

            # TODO Create visualization for html page SFD

            if "stock2" not in map_dict.keys():
                G = Network(height="800px",
                 width="800px",directed=True)
                for t in varlist:
                    G.add_node(t,label=str(t),shape='box', borderWidth=0, color='white')
                for kg, vg in map_dict.items():
                    if kg == 'stock1':
                        G.add_node(vg, shape='box', label=str(vg), borderWidth=3, bordercolor='black')
                        G.add_node("s1", shape='box', label=str('s1'), borderWidth=0, color='white')
                        G.add_node("e1", shape='box', label=str('e1'), borderWidth=0, color='white')


                for kkg, vvg in map_dict.items():
                    if kkg == "inflow1":
                        G.add_edge("s1", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow1':
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0], "e1",
                                   color='blue', label=vvg, border=3)
                with open(r"ModelsFormat\relationinCFD.txt") as f:
                    datafile = f.readlines()
                    for lin in datafile:
                        tempinvar = lin.split("=")[0]
                        tempstr = lin[lin.find("(") + 1:lin.find(")")]
                        templist = tempstr.split(',')
                        if tempstr != "":
                            for t in templist:
                                if t in varlist and tempinvar in varlist:
                                    G.add_edge(t, tempinvar, color='blue')

                                elif tempinvar != map_dict["stock1"] and tempinvar != map_dict["inflow1"] and tempinvar != map_dict["outflow1"] and t in varlist:

                                    G.add_edge(t, tempinvar, color='blue')

            elif "stock2" in map_dict.keys():
                G = Network(height="800px",
                 width="800px",directed=True)
                for t in varlist:
                    G.add_node(t, label=str(t), shape='box', borderWidth=0, color='white')
                for kg, vg in map_dict.items():
                    if kg == 'stock1':
                        G.add_node(vg, shape='box', label=str(vg), borderWidth=3, bordercolor='black')
                        G.add_node("s1", shape='box', label=str('s1'), borderWidth=0, color='white')
                        G.add_node("e1", shape='box', label=str('e1'), borderWidth=0, color='white')
                    if kg == 'stock2' and vg!="none":
                        G.add_node(vg, shape='triangle', label=str(vg))
                        G.add_node("s2", shape='box', label=str('s2'), borderWidth=0, color='white')
                        G.add_node("e2", shape='box', label=str('s2'), borderWidth=0, color='white')



                for kkg, vvg in map_dict.items():
                    if kkg == "inflow1":
                        G.add_edge("s1", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow1':
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0], "e1",
                                   color='blue', label=vvg, border=3)
                    if kkg == "inflow2" and vvg!="none":
                        G.add_edge("s2", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock2'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow2'and vvg!="none":
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock2'}.values())[0], "e2",
                                   color='blue', label=vvg, border=3)


                with open(r"ModelsFormat\relationinCFD.txt") as f:
                    datafile = f.readlines()
                    for lin in datafile:
                        tempinvar = lin.split("=")[0]
                        tempstr = lin[lin.find("(") + 1:lin.find(")")]
                        templist = tempstr.split(',')
                        if tempstr != "":
                            for t in templist:
                                if t in varlist and tempinvar in varlist:
                                    G.add_edge(t, tempinvar, color='blue')

                                elif tempinvar != map_dict["stock1"] and tempinvar != map_dict["stock2"] \
                                        and tempinvar != map_dict["inflow1"] and tempinvar != map_dict["outflow1"]\
                                        and tempinvar != map_dict["inflow2"] and tempinvar != map_dict["outflow2"] \
                                        and t in varlist:

                                    G.add_edge(t, tempinvar, color='blue')

            #G.save_graph(str(cwd)+"\\templates\mygraph.html")
            path = os.path.join('templates', 'mygraph.html')
            return

=======
import pandas as pd
from shutil import copyfile
import numpy as np
import os
from pyvis.network import Network
import matplotlib as plt
from collections import defaultdict
import os
class creat_CFD:

    def write_cfd(self,corr_df):
        cwd = os.getcwd()
        map_dict={}
        corr_df_dict = corr_df.to_dict('dict')

        tempvarnames = ['a','b','c','d','e','f','g','h','i','j']
        copyfile(os.path.join('ModelsFormat', 'testDFD.mdl'),os.path.join('ModelsFormat', "newtestDFD.mdl"))
        f = open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r')
        filedata = f.read()
        f.close()
        i = 0
        for kv in corr_df.index.values.tolist():
            filedata = filedata.replace(','+str(tempvarnames[i]+','), ','+str(kv)+',')
            f = open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'w')
            f.write(filedata)
            f.close()
            i +=1

        with open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                tempstr = ''
                for kmm, vmm in vm.items():
                    if vmm != 0 and tempstr == '':
                        tempstr = tempstr+str(kmm)
                    elif vmm != 0 and tempstr != '':
                        tempstr = tempstr + ',' + str(kmm)

                cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')


        with open(os.path.join('ModelsFormat', "newtestDFD.mdl"), 'r+') as f:
            content = f.read()
            tempplace= content.index(':L<%^E!@')
            f.seek(tempplace)
            f.write("")

        print ('done')

        #TODO: visualize CFD python

        corr_df[corr_df == 0] = 0
        G = Network(height="800px",
                 width="800px",directed=True)
        corr_dft = corr_df.T

        for act in corr_df.columns:
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
        G.save_graph(os.path.join('templates', 'mygraph.html'))
        #G.save_graph(str(os.getcwd())+"\\templates\mygraph.html")
        #G.save_graph(r"C:\Users\mahsa\PycharmProjects\SharedProjectForReviewCoopis2019\templates\cfdgraph.html")

        #TODO write the relation function into text for further use
        with open(os.path.join('ModelsFormat', "relationinCFD.txt"), 'w+') as cfdfile:
            cfdfile.seek(0)
            cfdfile.write('{UTF-8}\n')
            for km, vm in corr_df_dict.items():
                tempstr = ''
                for kmm, vmm in vm.items():
                    if vmm != 0 and tempstr == '':
                        tempstr = tempstr+str(kmm)
                    elif vmm != 0 and tempstr != '':
                        tempstr = tempstr + ',' + str(kmm)

                cfdfile.write( str(km) + '=' + 'A FUNCTION OF('+tempstr+')\n'+'~\n'+'~\n'+'|\n')


        return

    def read_cdf_2sfd_stockbased(self):
        cwd = os.getcwd()
        param_ele_dict = defaultdict(list)
        sd_df = pd.read_csv(str(cwd)+"\\static\images\SDLog2ShowInside.csv")
        sd_df.columns = sd_df.columns.str.replace(' ', '')
        if 10>len(sd_df.columns):
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)

                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
        if 16>len(sd_df.columns) >10:
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)
                    param_ele_dict['stock2'].append(col)
                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
                    param_ele_dict['inflow2'].append(col)
                    param_ele_dict['outflow2'].append(col)
            param_ele_dict['stock2'].append("none")
            param_ele_dict['inflow2'].append("none")
            param_ele_dict['outflow2'].append("none")

        if len(sd_df.columns) >16:
            for col in sd_df.columns:
                lower_col = col.lower()
                if "num" or "time" in lower_col:
                    param_ele_dict['stock1'].append(col)
                    param_ele_dict['stock2'].append(col)
                    param_ele_dict['stock3'].append(col)
                if "rate" in lower_col:
                    param_ele_dict['inflow1'].append(col)
                    param_ele_dict['outflow1'].append(col)
                    param_ele_dict['inflow2'].append(col)
                    param_ele_dict['outflow2'].append(col)
                    param_ele_dict['inflow3'].append(col)
                    param_ele_dict['outflow3'].append(col)
            param_ele_dict['inflow2'].append("none")
            param_ele_dict['outflow2'].append("none")
            param_ele_dict['stock2'].append("none")
            param_ele_dict['inflow3'].append("none")
            param_ele_dict['outflow3'].append("none")
            param_ele_dict['stock3'].append("none")

        return param_ele_dict

    def write_sfd_stockbased(self, map_dict):
        cwd = os.getcwd()
        if "stock2" in map_dict.keys():
            mainSFDfile = r"ModelsFormat\testDFD.mdl"
            newSFDfile = r"ModelsFormat\newtestDFD"

        else:
            mainSFDfile = r"ModelsFormat\1StockSFD.mdl"
            newSFDfile = r"ModelsFormat\new1StockSFD.mdl"


            tempvarnames = ['stock1', 'stock2', 'inflow1', 'outflow1', 'variable1', 'variable2', 'variable3',
                            'variable4', 'variable5', 'variable6']

            varlist = []
            with open(r"ModelsFormat\relationinCFD.txt") as fp:
                line = fp.readlines()
                for i in line:
                    if "=" in i:
                        varlist.append(i.split("=")[0])

            copyfile(mainSFDfile, newSFDfile)
            f = open(newSFDfile, 'r')
            filedata = f.read()
            f.close()
            i = 0
            for k, v in map_dict.items():
                if v in varlist:
                    varlist.pop(varlist.index(v))
                if v != 'submit':
                    filedata = filedata.replace(',' + str(k) + ',', ',' + str(v) + ',')
                    f = open(newSFDfile, 'w')
                    f.write(filedata)
                    f.close()
                i += 1

            for var in range(len(varlist)):
                filedata = filedata.replace(',' + "variable"+str(var+1) + ',', ',' + str(varlist[var]) + ',')
                f = open(newSFDfile, 'w')
                f.write(filedata)
                f.close()


            with open(r"ModelsFormat\relationinCFD.txt") as infile, open(newSFDfile, 'r+') as outfile:
                outfile.seek(0)
                for li in infile:
                    outfile.write(li)
                infile.close()
                outfile.close()

            # TODO Create visualization for html page SFD

            if "stock2" not in map_dict.keys():
                G = Network(height="800px",
                 width="800px",directed=True)
                for t in varlist:
                    G.add_node(t,label=str(t),shape='box', borderWidth=0, color='white')
                for kg, vg in map_dict.items():
                    if kg == 'stock1':
                        G.add_node(vg, shape='box', label=str(vg), borderWidth=3, bordercolor='black')
                        G.add_node("s1", shape='box', label=str('s1'), borderWidth=0, color='white')
                        G.add_node("e1", shape='box', label=str('e1'), borderWidth=0, color='white')


                for kkg, vvg in map_dict.items():
                    if kkg == "inflow1":
                        G.add_edge("s1", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow1':
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0], "e1",
                                   color='blue', label=vvg, border=3)
                with open(r"ModelsFormat\relationinCFD.txt") as f:
                    datafile = f.readlines()
                    for lin in datafile:
                        tempinvar = lin.split("=")[0]
                        tempstr = lin[lin.find("(") + 1:lin.find(")")]
                        templist = tempstr.split(',')
                        if tempstr != "":
                            for t in templist:
                                if t in varlist and tempinvar in varlist:
                                    G.add_edge(t, tempinvar, color='blue')

                                elif tempinvar != map_dict["stock1"] and tempinvar != map_dict["inflow1"] and tempinvar != map_dict["outflow1"] and t in varlist:

                                    G.add_edge(t, tempinvar, color='blue')

            elif "stock2" in map_dict.keys():
                G = Network(height="800px",
                 width="800px",directed=True)
                for t in varlist:
                    G.add_node(t, label=str(t), shape='box', borderWidth=0, color='white')
                for kg, vg in map_dict.items():
                    if kg == 'stock1':
                        G.add_node(vg, shape='box', label=str(vg), borderWidth=3, bordercolor='black')
                        G.add_node("s1", shape='box', label=str('s1'), borderWidth=0, color='white')
                        G.add_node("e1", shape='box', label=str('e1'), borderWidth=0, color='white')
                    if kg == 'stock2' and vg!="none":
                        G.add_node(vg, shape='triangle', label=str(vg))
                        G.add_node("s2", shape='box', label=str('s2'), borderWidth=0, color='white')
                        G.add_node("e2", shape='box', label=str('s2'), borderWidth=0, color='white')



                for kkg, vvg in map_dict.items():
                    if kkg == "inflow1":
                        G.add_edge("s1", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow1':
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock1'}.values())[0], "e1",
                                   color='blue', label=vvg, border=3)
                    if kkg == "inflow2" and vvg!="none":
                        G.add_edge("s2", list({kt: vt for kt, vt in map_dict.items() if kt == 'stock2'}.values())[0],
                                   color='blue', label=vvg, border=3)
                    if kkg == 'outflow2'and vvg!="none":
                        G.add_edge(list({kt: vt for kt, vt in map_dict.items() if kt == 'stock2'}.values())[0], "e2",
                                   color='blue', label=vvg, border=3)


                with open(r"ModelsFormat\relationinCFD.txt") as f:
                    datafile = f.readlines()
                    for lin in datafile:
                        tempinvar = lin.split("=")[0]
                        tempstr = lin[lin.find("(") + 1:lin.find(")")]
                        templist = tempstr.split(',')
                        if tempstr != "":
                            for t in templist:
                                if t in varlist and tempinvar in varlist:
                                    G.add_edge(t, tempinvar, color='blue')

                                elif tempinvar != map_dict["stock1"] and tempinvar != map_dict["stock2"] \
                                        and tempinvar != map_dict["inflow1"] and tempinvar != map_dict["outflow1"]\
                                        and tempinvar != map_dict["inflow2"] and tempinvar != map_dict["outflow2"] \
                                        and t in varlist:

                                    G.add_edge(t, tempinvar, color='blue')

            #G.save_graph(str(cwd)+"\\templates\mygraph.html")
            path = os.path.join('templates', 'mygraph.html')
            return

>>>>>>> ba9fe68d53340f50e0572b489d2912038af8c351
