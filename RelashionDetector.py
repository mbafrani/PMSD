import dcor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
import os
import seaborn as sns
from scipy.stats import pearsonr


class Relation_Detector():

    def only_correlation(self,data):
        data.fillna(data.mean(), inplace=True)
        only_corr = data.corr()
        return only_corr
    def show_corr_auto(self,sdLog,linrel,nonlinrel,ThreLin,ThreNLin,twshiftboolian):
        data = pd.read_csv(sdLog)
        data.fillna(data.mean(), inplace=True)
        dis_corr_df = pd.DataFrame(index=data.columns, columns=data.columns)
        twshiftboolian = int(twshiftboolian)

        for pr in data.columns:
            for dpr in data.columns:
                dis_corr_df[pr][dpr] = dcor.distance_correlation(data[pr], data[dpr])
                #dis_corr_df[pr][dpr] = scipy.spatial.distance.correlation(data[pr], data[dpr])
        dis_corr_df.fillna(data.mean(),inplace=True)
        allcorr = data.corr()
        allcorr = allcorr.filter(items=allcorr[allcorr > float(ThreLin)])
        allcorr = allcorr.filter(items=allcorr[allcorr<-1*float(ThreLin)])
        dis_corr_df = dis_corr_df.filter(items=dis_corr_df[dis_corr_df > float(ThreNLin)])
        dis_corr_df =dis_corr_df.filter(items=dis_corr_df[dis_corr_df <-1*float(ThreNLin)])
        allcorr_twshift = dis_corr_df.filter(items=dis_corr_df[dis_corr_df > float(ThreNLin)])
        dis_corr_df_twshift = dis_corr_df.filter(items=dis_corr_df[dis_corr_df <-1*float(ThreNLin)])
        #TODO correlation of each param with itself next timestep

        if twshiftboolian != 0:

                    for col in data.columns:
                        for ccol in data.columns:
                            shift_corr_dict = {}
                            shif_dis_corr_dict = {}
                            shift_corr_index_dict = {}
                            shif_dis_corr_index_dict ={}

                            for twshift in range(int(round(0.5 * len(data[data.columns[0]]),0))):
                                list2 = data[col].values
                                list2 = list2.tolist()
                                list1 = data[ccol].values
                                list1 = list1.tolist()

                                for i in range(int(twshift)):
                                    list2.pop(0)
                                    list1.pop()

                                p_corr_next, _ = pearsonr(list1, list2)
                                dis_corr_next = scipy.spatial.distance.correlation(list2, list1)
                                shift_corr_dict[twshift] = p_corr_next
                                shif_dis_corr_dict[twshift] = dis_corr_next

                            allcorr[col][ccol]= max(shift_corr_dict.values())
                            dis_corr_df[col][ccol] = max(shif_dis_corr_dict.values())
                            allcorr_twshift[col][ccol] = max(shift_corr_index_dict.keys())
                            dis_corr_df_twshift[col][ccol] = max(shif_dis_corr_index_dict.keys())

        allcorr.fillna(0,inplace=True)
        dis_corr_df.fillna(0,inplace=True)

        if linrel == 'on':
            fig = plt.figure()
            #fig.resize(50, 50)
            ax = fig.add_subplot(111)
            cax = ax.matshow(allcorr, cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(cax)
            ticks = np.arange(0, len(data.columns), 1)
            ax.set_xticks(ticks)

            ax.set_yticks(ticks)
            ax.set_xticklabels(data.columns,)
            ax.xaxis.set_ticks_position("bottom")
            plt.xticks(rotation=90)
            #plt.setp(ax.get_xticklabels(), rotation=45,va="center", rotation_mode="anchor")
            ax.set_yticklabels(data.columns)
            outpath= os.path.join("static","images","corr.png")
            plt.savefig(outpath,bbox_inches='tight')
            #plt.show()

        if nonlinrel == 'on':
            # todo show distance corrlation
            #plt.resize(50, 50)
            fig1 = plt.figure()
            sns.heatmap(dis_corr_df)
            outpath = os.path.join("static", "images", "discorr.png")
            plt.savefig(outpath, bbox_inches='tight')
            #plt.savefig('static/images/discorr.png',bbox_inches='tight')


        for pr in data.columns:
            relevelt_features = allcorr.index
            n_features =math.floor(math.sqrt(len(relevelt_features)))+1
            i = 1
            plt.figure()
            plt.subplot(n_features, n_features, i)
            plt.tight_layout()
            plt.suptitle(pr, size=8)
            plt.xlabel(str('Time'), fontsize=7)
            plt.ylabel(str(pr), fontsize=7)
            plt.bar(data.index, data[pr])
            i += 1
            plt.subplot(n_features, n_features, i)
            try:
                sns.distplot(data[pr])
            except:
                continue

            for ppr in relevelt_features:
                if ppr != pr and i<= n_features*n_features:
                    i += 1
                    plt.subplot(n_features , n_features, i)
                    plt.scatter(data[pr], data[ppr])
                    plt.xlabel(str(pr), fontsize=7)
                    plt.ylabel(str(ppr), fontsize=7)
                    #plt.title(str(pr))
                    plt.tight_layout()
            outputpath= os.path.join("static","images",str(pr.replace(" ",""))+'.png')
            plt.savefig(outputpath,bbox_inches='tight')
            #plt.show()

        print(dis_corr_df_twshift)

        return allcorr


