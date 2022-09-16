import os
import pm4py
import datetime
import re
from collections import defaultdict
from flask import Flask, render_template, request, session, flash, url_for, redirect, send_file
from flask import send_file, make_response
from flask_session import Session
from flask_cors import CORS, cross_origin


from SDSim.UIFinalVersion import Complete_sd
from SDSim.RelashionDetector import Relation_Detector
from SDSim.OrganizationalAspect import organization_aspect
from SDSim.writetomdlfile import creat_CFD
from SDSim.TimeSeriesSD import TW_Analysis
from SDSim.SimulationValidation import SimVal
import shutil

from integrated_framework.Interframwork import run

from integrated_framework.diagnostics.el import El
from integrated_framework.diagnostics.sdl import Sdl
from integrated_framework.diagnostics.relationdisc import *
from integrated_framework.diagnostics import ui_single_perspective
from SDSim.RelashionDetector import Relation_Detector
from SDSim.OrganizationalAspect import organization_aspect

cwd = os.getcwd()
app = Flask(__name__)

SECRET_KEY = "diagnostic-tooooool"
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)
CORS(app)

app.secret_key = SECRET_KEY
app.permanent_session_lifetime = datetime.timedelta(minutes=120) # Session only holds 120minutes
app.config['CACHE_TYPE'] = 'null'
app.config['UPLAOD_FOLDER'] = os.path.join(str(cwd), "static", "images")
app.config["FILE_UPLOADS"] = os.path.join(str(cwd), "static", "images")
app.config['SESSION_TYPE'] = SESSION_TYPE
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

ALLOWED_EXTENSIONS = set(['csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

createcfd = creat_CFD()
rel_sd = Relation_Detector()
org_asp = organization_aspect()
com_sd = Complete_sd()
global ready_EL






######DIAGNOSTICS######
@app.route('/BehaviorDiscovery.html')
@cross_origin(supports_credentials=True)
def get_behavior_discovery():
    if "SDLog" in session:
        sd_log = session['SDLog']
    else:
        sd_log = None
    return render_template('BehaviorDiscovery.html', sd_log=sd_log)


@app.route('/BehaviorDiscoveryResult.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_behavior_discovery_result():
    if request.method == 'POST':
        if "SDLog" in session:
            sd_log = session['SDLog']
        # else:
        if "submit start behavior" in request.form:
            sd_log_filename = request.files["SDLog"].filename
            if 'sd_log_filename' in session:
                sd_log_filename_session = session['sd_log_filename']
            else:
                sd_log_filename_session = 'new session'
            if sd_log_filename != sd_log_filename_session:
                # drop old sdl_obj when uploading new one
                if "sdl_obj" in session:
                    session.pop('sdl_obj')
                session['sd_log_filename'] = sd_log_filename
            if 'checked_points_single' in session:
                session.pop('checked_points_single')
        if 'checked_points_single' in session:
            checked_points = session['checked_points_single']
        else:
            checked_points = dict()
            checked_points['check_season'] = request.form.get("check_season")
            checked_points['cp_pelt'] = request.form.get("cp_pelt")
            checked_points['cp_bs'] = request.form.get("cp_bs")
            checked_points['ks_test'] = request.form.get("cp_ks")
            checked_points['granger'] = request.form.get("check_granger")
            checked_points['corr'] = request.form.get("check_corr")
            checked_points['sub_seq'] = request.form.get("sub_clustering")
            checked_points['uni_fore'] = request.form.get("uni_forecasting")
            checked_points['w_size'] = request.form.get("ThreWin")
            checked_points['stat_size'] = request.form.get("ThreStat")
            checked_points['forecasting'] = request.form.get("uni_forecasting")
            checked_points['forecast_n_period'] = request.form.get('forecast_n_period')
            session['checked_points_single'] = checked_points

        selected_aspect = request.form.get("selected_aspect")
        if 'sdl_obj' in session:
            sdl = session['sdl_obj']
        else:
            sd_log_filename = session['sd_log_filename']
            if 'el' in session:
                el = session['el']
                sdl = Sdl(path=os.path.join('Outputs', sd_log_filename), start_tp=el.get_earliest_timestamp())
            elif re.match('(\d{4})[/.-](\d{2})[/.-](\d{2}).*', request.form.get("datepicker")):
                date = pd.to_datetime(request.form.get('datepicker'))
                sdl = Sdl(path=os.path.join('Outputs', sd_log_filename), start_tp=date)
            else:
                sdl = Sdl(path=os.path.join('Outputs', sd_log_filename))
            session['sdl_obj'] = sdl
        params_list = sdl.columns
        if not selected_aspect:
            # select arrival rate as default aspect if none is choosen
            selected_aspect = params_list[0]
        res = ui_single_perspective.calc_res(sd_log=sdl, aspect=selected_aspect, checked_points=checked_points)

    return render_template('BehaviorDiscoveryResult.html', selected_aspect=selected_aspect, params_list=params_list,
                           res=res, raw_data=sdl.data.to_dict('list'), timed_data=sdl.timed_data)


@app.route('/MultipleDiscovery.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_multiple_perspective():
    sdLog1 = ''
    sdLog2 = ''
    sdl_1_col = []
    sdl_2_col = []

    if request.method == 'POST':
        if len(list(request.files.values())) == 2 and list(request.files.values())[0].filename != '':
            sdLog1 = request.files["SdLogOne"].filename
            sdLog2 = request.files["SdLogTwo"].filename
            # session['filename'] = x.filename
            # Upload as sdl object (expects csv in Outputs folder)
            sdl1 = Sdl(os.path.join("Outputs", sdLog1))
            sdl2 = Sdl(os.path.join("Outputs", sdLog2))

            sdl_1_col = sdl1.columns
            sdl_2_col = sdl2.columns

        elif list(request.form.keys())[0] == "aspect1":
            aspect1 = request.form.get("aspect1")
            aspect2 = request.form.get("aspect2")
    return render_template('MultipleDiscovery.html', sdl_1_col=sdl_1_col, sdl_2_col=sdl_2_col,
                           sdLog1_name=sdLog1, sdLog2_name=sdLog2)

@app.route('/MultipleDiscoveryResult.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_multiple_perspective_result():
    sdLog1 = ''
    sdLog2 = ''
    sdl_1_col = []
    sdl_2_col = []
    res = {}

    if request.method == 'POST':

        sdLog1 = request.files["SdLogOne"].filename
        sdLog2 = request.files["SdLogTwo"].filename

        checked_granger = request.form.get("check_granger")
        checked_corr = request.form.get("check_corr")
        checked_plot = request.form.get("check_plot")
        with_cp = request.form.get("check_plot_cp")
        res['corr_type'] = checked_corr
        res['granger_type'] = checked_granger
        res['checked_plot'] = checked_plot
        # session['filename'] = x.filename
        # Upload as sdl object (expects csv in Outputs folder)
        sdl1 = Sdl(os.path.join("Outputs", sdLog1))
        sdl2 = Sdl(os.path.join("Outputs", sdLog2))

        path_all_cp1 = os.path.join('static', 'images', 'all_cp_1.png')
        path_all_cp2 = os.path.join('static', 'images', 'all_cp_2.png')
        if checked_plot == 'on':
            if with_cp == 'on':
                sdl1.plot_all_with_cp(outputpath=path_all_cp1)
                sdl2.plot_all_with_cp(outputpath=path_all_cp2)
            else:
                sdl1.plot_all(outputpath=path_all_cp1)
                sdl2.plot_all(outputpath=path_all_cp2)


        path_granger = os.path.join('static', 'images', 'granger_2sdlogs.png')
        if checked_granger == 'granger_linear':
            grangers_causation_matrix_2sdlogs(sd_log_one=sdl1, sd_log_two=sdl2, plot=True, save_hm=True,
                                              outputpath=path_granger)
        elif checked_granger == 'granger_non_linear':
            non_linear_granger_causation(sd_log=sdl1, sd_log2=sdl2, plot=True, save_hm=True,
                                         outputpath=path_granger, maxlag=6)

        path_corr = os.path.join('static', 'images', 'pearson_2sdlogs.png')
        if checked_corr == 'pearson_corr':
            corr_pearson(sdl1, sdl2, plot=True, save_hm=True,
                         outputpath=path_corr)
        elif checked_corr == 'distance_corr':
            corr_distance_2sdLogs(sd_log1=sdl1, sd_log2=sdl2, plot=True, save_hm=True, outputpath=path_corr)

        sdl_1_col = sdl1.columns
        sdl_2_col = sdl2.columns

    return render_template('MultipleDiscoveryResult.html', sdl_1_col=sdl_1_col, sdl_2_col=sdl_2_col,
                           sdLog1_name=sdLog1, sdLog2_name=sdLog2, res=res,
                           img_cp1=path_all_cp1, img_cp2=path_all_cp2, img_corr=path_corr, img_granger=path_granger)


@app.route('/ProcessDiscoveryStart.html')
@cross_origin(supports_credentials=True)
def get_pd_start():
    return render_template('ProcessDiscoveryStart.html')


@app.route('/ProcessDiscovery.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def process_discovery():
    table = 'Activities'
    threshold = datetime.timedelta(minutes=0)
    if request.method == 'POST':
        table = request.form['table']
        try:
            threshold = datetime.timedelta(minutes=int(request.form['threshold']))
        except:
            threshold = datetime.timedelta(minutes=0)
    if "ready_event_log_path" in session:
        outputpath = session["ready_event_log_path"]
    else:
        flash('No Event Log loaded in this session. Please upload event Log firstly.', 'warning')
        #return redirect((url_for("get_pre_event_log")))
        return redirect((url_for("get_diagn_pre_event_log")))

    if session.get('el') is None:
        el = El(outputpath)
        session['el'] = el
    else:
        el = session['el']
    info = []
    info_bp_act = el.boxplot_act
    info_bp_trans = el.boxplot_trans
    info_bp_res = el.boxplot_res
    start_tp = el.get_earliest_timestamp()
    lifecycle = el.lifecycle
    act_recomm = el.act_recommendation
    trans_recomm = el.trans_recommendation
    res_dur = el.res_durations
    roles = el.res_recommendation
    dfg_img_path = os.path.join('static', 'images', 'dfg.png')
    gviz = pm4py.visualization.dfg.visualizer.apply(el.dfg_perf, log=el.log, soj_time=el.soj_time,
                                                    variant=pm4py.visualization.dfg.visualizer.Variants.PERFORMANCE)
    pm4py.visualization.dfg.visualizer.save(gviz, dfg_img_path)
    # bpmn_img_path = os.path.join('static', 'images', 'bpmn.png')
    # pm4py.save_vis_bpmn(el.bpmn_graph, bpmn_img_path)
    info.append(str(el.log_csv["Case ID"].nunique()))
    info.append(str(el.log_csv.shape[0]))
    info.append(str(el.log_csv["Activity"].nunique()))
    info.append(str(el.log_csv["Resource"].nunique()))
    info.append(start_tp)
    info.append(lifecycle)
    return render_template('ProcessDiscovery.html', img_path=dfg_img_path, act_recomm=act_recomm,
                           trans_recomm=trans_recomm, roles=roles, start_tp=start_tp, info=info, table=table,
                           threshold=threshold, info_bp_a=info_bp_act, info_bp_t=info_bp_trans, info_bp_r=info_bp_res,
                           res_dur=res_dur)
#######DIAGNOSTICS#######

@app.after_request
@cross_origin(supports_credentials=True)
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
@cross_origin(supports_credentials=True)
def get_first_page():
    session.clear()
    #return render_template('bootstrapTemplate.html')
    return render_template('GeneralTemplate.html')

@app.route('/bootstrapTemplate.html')
def get_sim_page():
   return render_template('bootstrapTemplate.html')

@app.route('/DESHomePage.html')
def get_des_page():
   return render_template('DESHomePage.html')


@app.route('/DiagnosticHomePage.html')
def get_diagnostic_page():
   return render_template('DiagnosticHomePage.html')

@app.route('/InsideEventLog.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_pre_event_log():
    com_sd = Complete_sd()
    event_log_cols = []
    event_log_cols_map = []
    el_info = ""
    download_file = ''
    response = ''
    first_msg = False
    sec_msg = False

    if request.method == 'POST':
        if len(list(request.files.values())) == 1 and list(request.files.values())[0].filename != '':
            x = request.files["Event Log"]
            session['filename'] = x.filename
            if 'csv' in x.filename:
                outputpath = os.path.join("Outputs", "ready_event_log.csv")
                x.save(outputpath)
                event_log, event_log_cols = com_sd.get_input_file(outputpath)
            elif 'xes' in x.filename:
                event_log, event_log_cols = com_sd.get_input_file(x.filename)
                event_log.to_csv(os.path.join("Outputs", "ready_event_log.csv"))
            first_msg = True
        elif list(request.form.keys())[0] == "CaseID":
            outputpath = os.path.join("Outputs", "ready_event_log.csv")
            event_log, event_log_cols = com_sd.get_input_file(outputpath)
            event_log_cols_map.append(request.form.get("CaseID"))
            event_log_cols_map.append(request.form.get("Activity"))
            event_log_cols_map.append(request.form.get("Resource"))
            event_log_cols_map.append(request.form.get("StartTime"))
            event_log_cols_map.append(request.form.get("CompleteTime"))
            event_log_ready = com_sd.add_needed_column(event_log, event_log_cols_map)
            outputpath = os.path.join("Outputs", "ready_event_log.csv")

            # copy_to('/local/foo.txt', 'my-container:/tmp/foo.txt')
            event_log_ready.to_csv(outputpath, columns=event_log_ready.columns)
            session.permanent = True
            session["ready_event_log_path"] = outputpath
            event_log = event_log_ready
            matrix= org_asp.create_matrix(event_log)
            org_asp.create_DFG(matrix)
            # download_file = send_file(r'Output\ready_event_log.csv',  mimetype='csv',attachment_filename='ready_event_log.csv', as_attachment=False)
            el_info = el_info + "Number of Cases:" + str(event_log["Case ID"].nunique()) + "\n Number of Events:" + str(
                event_log.shape[0])
            download_file = outputpath
            ready_EL = event_log
            first_msg = False
            sec_msg = True
    return download_file, render_template('/InsideEventLog.html', el_cols=event_log_cols, el_info=el_info,
                                          first_msg=first_msg, sec_msg=sec_msg)


@app.route('/InsideEventLogDiagnostic.html', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_diagn_pre_event_log():
    com_sd = Complete_sd()
    event_log_cols = []
    event_log_cols_map = []
    el_info = ""
    download_file = ''
    response = ''
    first_msg = False
    sec_msg = False

    if request.method == 'POST':
        if len(list(request.files.values())) == 1 and list(request.files.values())[0].filename != '':
            x = request.files["Event Log"]
            session['filename'] = x.filename
            if 'csv' in x.filename:
                outputpath = os.path.join("Outputs", "ready_event_log.csv")
                x.save(outputpath)
                event_log, event_log_cols = com_sd.get_input_file(outputpath)
            elif 'xes' in x.filename:
                event_log, event_log_cols = com_sd.get_input_file(x.filename)
                event_log.to_csv(os.path.join("Outputs", "ready_event_log.csv"))
            first_msg = True
        elif list(request.form.keys())[0] == "CaseID":
            outputpath = os.path.join("Outputs", "ready_event_log.csv")
            event_log, event_log_cols = com_sd.get_input_file(outputpath)
            event_log_cols_map.append(request.form.get("CaseID"))
            event_log_cols_map.append(request.form.get("Activity"))
            event_log_cols_map.append(request.form.get("Resource"))
            event_log_cols_map.append(request.form.get("StartTime"))
            event_log_cols_map.append(request.form.get("CompleteTime"))
            event_log_ready = com_sd.add_needed_column(event_log, event_log_cols_map)
            outputpath = os.path.join("Outputs", "ready_event_log.csv")

            # copy_to('/local/foo.txt', 'my-container:/tmp/foo.txt')
            event_log_ready.to_csv(outputpath, columns=event_log_ready.columns)
            session.permanent = True
            session["ready_event_log_path"] = outputpath
            event_log = event_log_ready
            # matrix= org_asp.create_matrix(event_log)
            # org_asp.create_DFG(matrix)
            # download_file = send_file(r'Output\ready_event_log.csv',  mimetype='csv',attachment_filename='ready_event_log.csv', as_attachment=False)
            el_info = el_info + "Number of Cases:" + str(event_log["Case ID"].nunique()) + "\n Number of Events:" + str(
                event_log.shape[0])
            download_file = outputpath
            ready_EL = event_log
            first_msg = False
            sec_msg = True
    return download_file, render_template('/InsideEventLogDiagnostic.html', el_cols=event_log_cols, el_info=el_info,
                                          first_msg=first_msg, sec_msg=sec_msg)

@app.route('/downloadlog')
@cross_origin(supports_credentials=True)
def ready_event_log():
    cwd = os.getcwd()
    event_log_path= os.path.join(cwd,"Outputs","ready_event_log.csv")
    return send_file(event_log_path,as_attachment=True)

@app.route('/EventLog.html')
@cross_origin(supports_credentials=True)
def get_event_log():

    return render_template('EventLog.html')

@app.route('/mygraph.html')
@cross_origin(supports_credentials=True)
def get_CFD():
   return render_template('mygraph.html')
@app.route('/mygraph1.html')
@cross_origin(supports_credentials=True)
def get_auto_CFD():
   return render_template('mygraph1.html')

@app.route('/EventLogResult.html', methods = ['POST', 'GET'])
@cross_origin(supports_credentials=True)
def result():
    com_sd = Complete_sd()
    generated_SD_log = []
    act_list = []
    aspect=""
    time_window=[]
    inactive='off'
    if request.method == 'POST':
        try:
            event_log_address = request.form["Event Log"]
            event_log = com_sd.get_input_file(os.path.join("Outputs",event_log_address))

        except:
            event_log = com_sd.get_input_file((event_log_address))

        time_window.append(request.form["time_window"])

        if "general" in request.form.keys():
            aspect = request.form["general"]
        else:
            aspect=""
        inactive = request.form.get("inactive")
        if inactive!="on":
            if aspect =="General":
                generated_SD_log.append((com_sd.TW_discovery_process_calculation_twlist(time_window, event_log[0],aspect))[0])
            if aspect =='Organizational':
                resource_freq_df = org_asp.find_resource(event_log[0])
                _,org_list = org_asp.find_roles(resource_freq_df)
                for org in list(org_list.keys()):
                    filtered_log = org_asp.filter_log_org(event_log[0],org)
                    if len(filtered_log['Case ID']) >1:
                        generated_SD_log.append((com_sd.TW_discovery_process_calculation_twlist(time_window,filtered_log,org))[0])

            if aspect =='Activity Flow':
                act_list = event_log[0]['Activity'].unique().tolist()
                for act in act_list:
                    features_list = com_sd.create_features_name ("Activities",act)
                    generated_SD_log.append(str(com_sd.select_features(features_list, event_log[0], time_window,"Activities")))
                """
                for act in act_list:
                    act_list_filter = []
                    act_list_filter.append(act)
                    filtered_log = org_asp.filter_log_act(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        generated_SD_log.append(com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, act))
                """
            if aspect =='Resources':
                act_list = event_log[0]['Resource'].unique().tolist()
                for act in act_list:
                    features_list = com_sd.create_features_name ("Resources",act)
                    generated_SD_log.append((com_sd.select_features(features_list, event_log[0], time_window,"Resources")))
                """
                for act in act_list:
                    act_list_filter = []
                    act_list_filter.append(act)
                    filtered_log = org_asp.filter_log_res(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        generated_SD_log.append((com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, act))[0])
                """
            elif request.form["AcReList"] !='':
                acreslist=request.form["AcReList"].split(",")
                acreslist=[(i.replace("_","")).replace(" ","") for i in acreslist]
                res_list = event_log[0]['Resource'].unique().tolist()
                act_list = event_log[0]['Activity'].unique().tolist()
                if acreslist[0] in res_list:
                    act_list_filter = acreslist
                    filtered_log = org_asp.filter_log_res(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        generated_SD_log.append((com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, request.form["AcReList"]))[0])
                if acreslist[0] in act_list:
                    act_list_filter = acreslist
                    filtered_log = org_asp.filter_log_act(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        generated_SD_log.append(
                            (com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, request.form["AcReList"]))[0])
        else:
            if aspect =="General":
                tempsdlog=(com_sd.TW_discovery_process_calculation_twlist(time_window, event_log[0],aspect))[1]
                generated_SD_log.append(com_sd.Post_process_tw(tempsdlog,aspect))

            if aspect =='Organizational':
                resource_freq_df = org_asp.find_resource(event_log[0])
                _,org_list = org_asp.find_roles(resource_freq_df)
                for org in list(org_list.keys()):
                    filtered_log = org_asp.filter_log_org(event_log[0],org)
                    if len(filtered_log['Case ID']) >1:
                        tempsdlog=(com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, org))[1]
                        generated_SD_log.append(com_sd.Post_process_tw(tempsdlog,org))

            if aspect =='Activity Flow':
                act_list = event_log[0]['Activity'].unique().tolist()
                for act in act_list:
                    features_list = com_sd.create_features_name ("Activities",act)
                    generated_SD_log.append(com_sd.select_features(features_list, event_log, time_window,"Activities"))

            if aspect == 'Resources':
                act_list = event_log[0]['Resource'].unique().tolist()
                for act in act_list:
                    act_list_filter = []
                    act_list_filter.append(act)
                    filtered_log = org_asp.filter_log_res(event_log[0], act_list_filter)
                    if len(filtered_log['Case ID']) > 1:
                        tempsdlog = (com_sd.TW_discovery_process_calculation_twlist(time_window, filtered_log, act))[1]
                        generated_SD_log.append(com_sd.Post_process_tw(tempsdlog, act))

    return render_template("EventLogResult.html",sd_log = generated_SD_log,aspect= aspect,act_list = act_list)

@app.route('/downloadSDlog', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def ready_sd_log():
    shutil.make_archive("Outputs", 'zip', "Outputs")
    SD_log_path = os.path.join("Outputs.zip")
    return send_file(SD_log_path,as_attachment=True)


@app.route('/SDLog.html',methods = ['POST', 'GET'])
@cross_origin(supports_credentials=True)
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
@cross_origin(supports_credentials=True)
def get_SD_Resutl():
    try:
        corr_df = pd.read_csv(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"))
    except:
        pass
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
@cross_origin(supports_credentials=True)
def get_mapping():
    createcfd = creat_CFD()
    rel_det = Relation_Detector()
    data = pd.read_csv(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"))
    #run(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"),cyclefree)
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
@cross_origin(supports_credentials=True)
def map_param_ele():
    params_list = createcfd.read_cdf_2sfd_stockbased()
    if request.method =='POST':
        createcfd.write_sfd_stockbased(request.values)

    return  render_template('DesignedSFD.html',param_list=params_list)


@app.route('/AutoDesignedCDF.html',methods= ['GET','POST'])
@cross_origin(supports_credentials=True)
def Auto_DesignedCLD():

    if request.method =='POST':
        sd_log = request.form["SDLog"]
        cyclefree = 0
        try:
            cyclefree=request.form["general"]
            cyclefree =1
        except:
            pass
        try:
            run(sd_log,cyclefree)
        except:
            run(os.path.join(str(cwd),'Outputs',sd_log),cyclefree)
        #run(os.path.join(str(cwd),"static","images","SDLog2ShowInside.csv"))

    return  render_template('AutoDesignedCDF.html')

@app.route('/downloadCLDModel', methods=['GET', 'POST'])
def downloadCLDModel():
    CLD_Model_path = os.path.join("ModelsFormat","newtestDFD.mdl")
    return send_file(CLD_Model_path,as_attachment=True)

@app.route('/downloadSFDModel', methods=['GET', 'POST'])
def downloadSFDModel():
    SFD_Model_path = os.path.join("ModelsFormat","new1StockSFD.mdl")
    return send_file(SFD_Model_path,as_attachment=True)

@app.route('/Validation.html', methods = ['GET','POST'])
@cross_origin(supports_credentials=True)
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
@cross_origin(supports_credentials=True)
def Stability_TW_Test():
    twa = TW_Analysis()
    time_window_list = []
    TW_Dete_dict = {}
    aspect = ""
    tw_result=""
    if request.method == 'POST':
        event_log_address= request.form["Event Log"]
        try:
            event_log = com_sd.get_input_file(os.path.join(event_log_address))
        except:
            event_log = com_sd.get_input_file(os.path.join("Outputs", event_log_address))
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
    port = int(os.environ.get('PORT', 5000))
    print(app.secret_key)
    app.run(debug = True,host = "0.0.0.0",port=port)