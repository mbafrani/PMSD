
import os
from collections import defaultdict
from flask import Flask, render_template, request, url_for,redirect
import pandas as pd
from flask import send_file,make_response
from UIFinalVersion import Complete_sd
from RelashionDetector import Relation_Detector
from OrganizationalAspect import organization_aspect
from writetomdlfile import creat_CFD
from TimeSeriesSD import TW_Analysis
from SimulationValidation import SimVal

cwd = os.getcwd()
app = Flask(__name__)
app.config['CACHE_TYPE']='null'
app.config['UPLAOD_FOLDER']=os.path.join(str(cwd),"static","images")
app.config["FILE_UPLOADS"] = os.path.join(str(cwd),"static","images")
ALLOWED_EXTENSIONS = set(['csv','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['SESSION_TYPE'] = 'filesystem'
createcfd = creat_CFD()
rel_sd = Relation_Detector()
org_asp = organization_aspect()
com_sd = Complete_sd()

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
def get_first_page():
   return render_template('bootstrapTemplate.html')

@app.route('/InsideEventLog.html', methods=['POST','GET'])
def get_pre_event_log():
    com_sd = Complete_sd()
    event_log_cols =[]
    event_log_cols_map= []
    el_info= ""
    download_file=''
    response=''
    if request.method == 'POST':
        if len(list(request.files.values())) ==1 and list(request.files.values())[0].filename != '':
            x = request.files["Event Log"]
            outputpath=os.path.join("Outputs","ready_event_log.csv")
            x.save(outputpath)
            event_log, event_log_cols = com_sd.get_input_file(outputpath)
            el_info = el_info +"Number of Cases:"+str(event_log[event_log.columns[0]].nunique())+"\n Number of Events:"+ str(event_log.shape[0])
        elif list(request.form.keys())[1]=="CaseID" :
            outputpath = os.path.join("Outputs", "ready_event_log.csv")
            event_log, event_log_cols = com_sd.get_input_file(outputpath)
            event_log_cols_map.append(request.form.get("CaseID"))
            event_log_cols_map.append(request.form.get("Activity"))
            event_log_cols_map.append(request.form.get("Resource"))
            event_log_cols_map.append(request.form.get("StartTime"))
            event_log_cols_map.append(request.form.get("CompleteTime"))
            event_log_ready = com_sd.add_needed_column(event_log,event_log_cols_map)
            outputpath = os.path.join("Outputs", "ready_event_log.csv")
            event_log_ready.to_csv(outputpath,columns=event_log_ready.columns)
            event_log=event_log_ready
            matrix= org_asp.create_matrix(event_log)
            org_asp.create_DFG(matrix)
            #download_file = send_file(r'Output\ready_event_log.csv',  mimetype='csv',attachment_filename='ready_event_log.csv', as_attachment=False)
            download_file = outputpath

    return  download_file, render_template('InsideEventLog.html', el_cols=event_log_cols,el_info =el_info )

@app.route('/EventLog.html')
def get_event_log():

    return render_template('EventLog.html')

@app.route('/mygraph.html')
def get_CFD():
   return render_template('mygraph.html')

@app.route('/EventLogResult.html', methods = ['POST', 'GET'])
def result():
    com_sd = Complete_sd()
    generated_SD_log = []
    act_list = []
    aspect=""
    time_window=[]
    inactive='off'
    if request.method == 'POST':
        event_log_address = request.form["Event Log"]
        event_log = com_sd.get_input_file(event_log_address)
        time_window.append(request.form["time_window"])
        aspect = request.form["general"]
        inactive = request.form.get("inactive")
        if inactive!="on":
            if aspect =="General":
                generated_SD_log.append(com_sd.TW_discovery_process_calculation_twlist(time_window, event_log[0],aspect))
            if aspect =='Organizational':
                resource_freq_df = org_asp.find_resource(event_log[0])
                _,org_list = org_asp.find_roles(resource_freq_df)
                for org in list(org_list.keys()):
                    filtered_log = org_asp.filter_log_org(event_log[0],org)
                    if len(filtered_log['Case ID']) >1:
                        generated_SD_log.append(com_sd.TW_discovery_process_calculation_twlist(time_window,filtered_log,org))
            if aspect =='Activity Flow':
                act_list = event_log[0]['Activity'].unique().tolist()
                for act in act_list:
                    act_list_filter = []
                    act_list_filter.append(act)
                    filtered_log = org_asp.filter_log_act(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        generated_SD_log.append(com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, act))

        else:
            if aspect =="General":
                tempsdlog=com_sd.TW_discovery_process_calculation_twlist(time_window, event_log[0],aspect)
                generated_SD_log.append(com_sd.Post_process_tw(tempsdlog,aspect))

            if aspect =='Organizational':
                resource_freq_df = org_asp.find_resource(event_log[0])
                _,org_list = org_asp.find_roles(resource_freq_df)
                for org in list(org_list.keys()):
                    filtered_log = org_asp.filter_log_org(event_log[0],org)
                    if len(filtered_log['Case ID']) >1:
                        tempsdlog=com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, org)
                        generated_SD_log.append(com_sd.Post_process_tw(tempsdlog,org))
            if aspect =='Activity Flow':
                act_list = event_log[0]['Activity'].unique().tolist()
                for act in act_list:
                    act_list_filter = []
                    act_list_filter.append(act)
                    filtered_log = org_asp.filter_log_act(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID'])>1:
                        tempsdlog=com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, act)
                        generated_SD_log.append(com_sd.Post_process_tw(tempsdlog,act))



    return render_template("EventLogResult.html",sd_log = generated_SD_log,aspect= aspect,act_list = act_list)

@app.route('/SDLog.html',methods = ['POST', 'GET'])
def get_SD_log():
    rel_sd = Relation_Detector()

    if request.method == 'POST':
        SD_log_address = request.files["SDLog"]
        Corr_Lin = request.form.get("Linear")
        Corr_NLin = request.form.get("NLinear")
        Corr_Lin_Thre = request.form.get("ThreLinear")
        Corr_NLin_Thre = request.form.get("ThreNLinear")
        TWshift = request.form.get("TWshift")
        allcorr = rel_sd.show_corr_auto(os.path.join('Outputs',SD_log_address.filename ),Corr_Lin,Corr_NLin,Corr_Lin_Thre,Corr_NLin_Thre,TWshift)
        params_list = allcorr.columns
        if SD_log_address:
            temp = pd.read_csv(os.path.join('Outputs',SD_log_address.filename ))
            temp.to_csv(os.path.join(
                str(cwd),"static","images","SDLog2ShowInside.csv"), index=False)
            request.files["SDLog"].save(os.path.join(app.config["FILE_UPLOADS"], SD_log_address.filename))

    return render_template('SDLog.html')

@app.route('/SDLogResutl.html',methods = ['POST', 'GET'])
def get_SD_Resutl():
    corr_df = pd.read_csv(os.path.join("static","images","SDLog2ShowInside.csv"))
    corr_df.columns = corr_df.columns.str.replace(' ', '')
    corr_df = rel_sd.only_correlation(corr_df)
    params_list = corr_df.columns
    param_img_name = params_list[0]
    if request.method == 'POST':
        param_img_names = request.form.get("params")
        if param_img_names !='submit':
            param_img_name = param_img_names
    return render_template('SDLogResutl.html', params_list=params_list, param_name = str(param_img_name)+".png")

@app.route('/DesignedCDF.html',methods = ['GET','POST'])
def get_mapping():
    createcfd = creat_CFD()
    rel_det = Relation_Detector()
    data = pd.read_csv(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"))
    data.columns = data.columns.str.replace(' ', '')
    corr_df =rel_det.only_correlation(data)
    corr_df_sign = rel_det.only_correlation(data)
    corr_df_sign=corr_df_sign.fillna(0)
    corr_df_sign.fillna(0)
    params_list = corr_df.to_dict('dict')
    user_corr_df=pd.DataFrame(columns=corr_df.columns,index=corr_df.columns)
    if request.method =='POST':
        for chk, chv in request.form.items():
            if chk != 'submit':
                if chk.split('+')[1] != "No":
                    if corr_df_sign[chk.split('+')[0]][chk.split('+')[1]]==0:
                        user_corr_df[chk.split('+')[0]][chk.split('+')[1]] = 1
                    else:
                        user_corr_df[chk.split('+')[0]][chk.split('+')[1]] = corr_df_sign[chk.split('+')[0]][chk.split('+')[1]]

                else:
                    user_corr_df[chk.split('+')[0]].values[:]=0

        user_corr_df.fillna(0,inplace=True)
        createcfd.write_cfd(user_corr_df)
    return render_template('DesignedCDF.html',param_list=params_list)

@app.route('/DesignedSFD.html',methods= ['GET','POST'])
def map_param_ele():
    params_list = createcfd.read_cdf_2sfd_stockbased()
    if request.method =='POST':
        createcfd.write_sfd_stockbased(request.values)

    return  render_template('DesignedSFD.html',param_list=params_list)

@app.route('/Validation.html', methods = ['GET','POST'])
def validation_sd_sim():
    val_image_names = []
    simval = SimVal()
    if request.method == "POST":
        real_data = request.files["SDLog"]
        headers = {"Content-Disposition": "attachment; filename=%s" % real_data.filename}
        with open(os.path.join('Outputs',real_data.filename ), 'r') as f:
            body = f.read()
        make_response((body, headers))

        sd_model = request.files["SDResultLog"]
        sim_res = simval.read_model(os.path.join('ModelsFormat',sd_model.filename ), os.path.join('Outputs',real_data.filename ))
        res_sim_dict = simval.creat_real_sim_dict(real_data, sim_res)
        val_image_names=simval.validate_results(res_sim_dict)
    return render_template('Validation.html', val_image_names=val_image_names)


@app.route('/TimeTest.html', methods = ['GET','POST'])
def Stability_TW_Test():
    twa = TW_Analysis()
    time_window_list = []
    TW_Dete_dict = {}
    aspect = ""
    tw_result=""
    if request.method == 'POST':
        event_log_address= request.form["Event Log"]
        event_log = com_sd.get_input_file(event_log_address)
        time_window_list.append(request.form["Hourly"])
        time_window_list.append(request.form["Daily"])
        time_window_list.append(request.form["Weekly"])
        time_window_list.append(request.form["Monthly"])
        time_window_list = [i for i in time_window_list if i]
        aspect = request.form["general"]

        if aspect == "General":
            tw_discovered = twa.TW_discovery_process_calculation_twlist(event_log[0], time_window_list, aspect)
            generated_SD_log = tw_discovered[1]
            overall_dict =tw_discovered[0]
            TW_Dete_dict= twa.Detect_pattern_tw(overall_dict, event_log[0],'all')
            tw_best = max(TW_Dete_dict, key=TW_Dete_dict.get)
            tw_worse = min(TW_Dete_dict, key=TW_Dete_dict.get)
            for k,v in TW_Dete_dict.items():
                if abs(v[2])>0.5:
                    tw_result = tw_result+' '+str(k)+ ' is a strong pattern.'
                else:
                    tw_result = tw_result + ' ' + str(k) + ' is not a strong pattern.'


            tw_result = tw_result+"\n The Strongest Pattern Discovered: "+str(abs(TW_Dete_dict.get(tw_best)[1]))+" "+str(tw_best)+"\n"
                        #"\n The Weakest Pattern: "+str(abs(TW_Dete_dict.get(tw_worse)[1]))+' For ' +str(tw_worse)

            active_overall_dict = {}
            active_overall_dict["Arrival rate"]={}
            for sdlog_name in generated_SD_log:
                sd_log =pd.read_csv(os.path.join("Outputs",sdlog_name))
                name_sd_tw = sdlog_name.split("_")[1]
                overall_dict["Arrival rate"].get(name_sd_tw)
                Active_SD_Log= (twa.Post_process_tw(sd_log, TW_Dete_dict))
                active_overall_dict["Arrival rate"][name_sd_tw]= Active_SD_Log[Active_SD_Log.columns[0]]

                active_TW_Dete_dict = twa.Detect_pattern_tw(active_overall_dict, event_log[0],'active')

            twa.Detect_best_user_tw(active_TW_Dete_dict,active_overall_dict)
            #twa.Detect_best_tw(TW_Dete_dict, overall_dict)
            #twa.Detect_best_tw_all_Windows(TW_Dete_dict, overall_dict)

    return render_template('TimeTest.html',tw_result = tw_result)


if __name__ == '__main__':
   app.run(debug = True)