from django.shortcuts import render
import os
import pickle
import numpy as np
import requests
from datetime import datetime
from django.http import HttpResponse
from django.http import JsonResponse
import json
import joblib


def index(request):
    context = {
        'konten': 'snippets/homeKonten.html',
        'hasil': "",
        'water': 'none',
        'dateNow': 'none',
        'hasilRealtime': 'none',
    }
    if request.method == 'POST':
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, 'model.pkl')
        f = open(file_path, 'rb')
        model = pickle.load(f)
        f.close()
        context['level'] = request.POST['level']
        context['bulan'] = request.POST['bulan']
        y_pred = model.predict(
            np.array([[request.POST['level'], request.POST['bulan']]]))
        if y_pred < 0.5:
            context['hasil'] = "Tidak Berpotensi Banjir ROB"
        else:
            context['hasil'] = "Berpotensi Banjir ROB"

    return render(request, './index.html', context)


def jst(request):
    context = {
        'konten': 'snippets/jstKonten.html',
    }
    return render(request, 'index.html', context)


def coba(request):

    context = {}

    if request.method == 'GET':
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, 'model.pkl')
        f = open(file_path, 'rb')
        model = pickle.load(f)
        f.close()
        y_pred = model.predict(
            np.array([[request.GET['level'], request.GET['bulan']]]))
        if y_pred < 0.5:
            context['hasilreal'] = "Tidak Berpotensi Banjir ROB"
            context['level'] = request.GET['level']
            context['bulan'] = request.GET['bulan']
        else:
            context['hasilreal'] = "Berpotensi Banjir ROB"
            context['level'] = request.GET['level']
            context['bulan'] = request.GET['bulan']
    return JsonResponse(context)


def lin(request):

    context = {}

    if request.method == 'GET':
        module_dir = os.path.dirname(__file__)
        pathP = os.path.join(module_dir, './model_Lin/pressure_linear.pkl')
        pathT = os.path.join(module_dir, './model_Lin/temp_linear.pkl')
        pathRH = os.path.join(module_dir, './model_Lin/rh_linear.pkl')
        pathWS = os.path.join(module_dir, './model_Lin/windspeed_linear.pkl')
        pathWD = os.path.join(module_dir, './model_Lin/winddir_linear.pkl')
        pathRN = os.path.join(module_dir, './model_Lin/rain_linear.pkl')
        pathWL = os.path.join(module_dir, './model_Lin/waterlevel_linear.pkl')
        pathWT = os.path.join(module_dir, './model_Lin/watertemp_linear.pkl')
        pathSR = os.path.join(module_dir, './model_Lin/solrad_linear.pkl')
        modelP = joblib.load(pathP)
        modelT = joblib.load(pathT)
        modelRH = joblib.load(pathRH)
        modelWS = joblib.load(pathWS)
        modelWD = joblib.load(pathWD)
        modelRN = joblib.load(pathRN)
        modelWL = joblib.load(pathWL)
        modelWT = joblib.load(pathWT)
        modelSR = joblib.load(pathSR)

        prediksiP = modelP.predict([request.GET['p'].split(',')])[0]
        prediksiT = modelT.predict([request.GET['t'].split(',')])[0]
        prediksiRH = modelRH.predict([request.GET['rh'].split(',')])[0]
        prediksiWS = modelWS.predict([request.GET['ws'].split(',')])[0]
        prediksiWD = modelWD.predict([request.GET['wd'].split(',')])[0]
        prediksiRN = modelRN.predict([request.GET['rn'].split(',')])[0]
        prediksiWL = modelWL.predict([request.GET['wl'].split(',')])[0]
        prediksiWT = modelWT.predict([request.GET['wt'].split(',')])[0]
        prediksiSR = modelSR.predict([[float(i) for i in request.GET['sr'].split(',')]])[0]

        # pressure
        if prediksiP > 1014:
            context['prediksiP'] = "Over"
        elif prediksiP > 1002:
            context['prediksiP'] = "Normal"
        else:
            context['prediksiP'] = "Warning"

        # temp
        if prediksiT > 37.92:
            context['prediksiT'] = "Over"
        elif prediksiT > 17.72:
            context['prediksiT'] = "Normal"
        else:
            context['prediksiT'] = "Warning"
        
        # RH
        if prediksiRH > 24.52:
            context['prediksiRH'] = "Normal"
        else:
            context['prediksiRH'] = "Warning"

        # wind speed
        if prediksiWS > 5.85:
            context['prediksiWS'] = "Over"
        elif prediksiWS > 3.6:
            context['prediksiWS'] = "Warning"
        else:
            context['prediksiWS'] = "Normal"

        # Wind Dir
        if prediksiWD > 17:
            context['prediksiWD'] = "Over"
        elif prediksiWD > 16.9:
            context['prediksiWD'] = "Warning"
        else:
            context['prediksiWD'] = "Normal"

        # Rain
        if prediksiRN > 10:
            context['prediksiRN'] = "Over"
        elif prediksiP > 9.5:
            context['prediksiRN'] = "Warning"
        else:
            context['prediksiRN'] = "Normal"

        # Water Level
        if prediksiWL > 4:
            context['prediksiWL'] = "Warning"
        else:
            context['prediksiWL'] = "Normal"
        
        # Water TEmp
        if prediksiWT > 35:
            context['prediksiWT'] = "Warning"
        else:
            context['prediksiWT'] = "Normal"

        # Solar Rad
        if prediksiSR > 590:
            context['prediksiSR'] = "Over"
        elif prediksiP > 588.4:
            context['prediksiSR'] = "Warning"
        else:
            context['prediksiSR'] = "Normal"

    return JsonResponse(context)


def rbf(request):

    context = {}

    if request.method == 'GET':
        module_dir = os.path.dirname(__file__)
        pathP = os.path.join(module_dir, './model_rbf/pressure_rbf.pkl')
        pathT = os.path.join(module_dir, './model_rbf/temp_rbf.pkl')
        pathRH = os.path.join(module_dir, './model_rbf/rh_rbf.pkl')
        pathWS = os.path.join(module_dir, './model_rbf/windspeed_rbf.pkl')
        pathWD = os.path.join(module_dir, './model_rbf/winddir_rbf.pkl')
        pathRN = os.path.join(module_dir, './model_rbf/rain_rbf.pkl')
        pathWL = os.path.join(module_dir, './model_rbf/waterlevel_rbf.pkl')
        pathWT = os.path.join(module_dir, './model_rbf/watertemp_rbf.pkl')
        # pathSR = os.path.join(module_dir, './model_rbf/solrad_linear.pkl')
        modelP = joblib.load(pathP)
        modelT = joblib.load(pathT)
        modelRH = joblib.load(pathRH)
        modelWS = joblib.load(pathWS)
        modelWD = joblib.load(pathWD)
        modelRN = joblib.load(pathRN)
        modelWL = joblib.load(pathWL)
        modelWT = joblib.load(pathWT)
        # modelSR = joblib.load(pathSR)

        prediksiP = modelP.predict([request.GET['p'].split(',')])[0]
        prediksiT = modelT.predict([request.GET['t'].split(',')])[0]
        prediksiRH = modelRH.predict([request.GET['rh'].split(',')])[0]
        prediksiWS = modelWS.predict([request.GET['ws'].split(',')])[0]
        prediksiWD = modelWD.predict([request.GET['wd'].split(',')])[0]
        prediksiRN = modelRN.predict([request.GET['rn'].split(',')])[0]
        prediksiWL = modelWL.predict([request.GET['wl'].split(',')])[0]
        prediksiWT = modelWT.predict([request.GET['wt'].split(',')])[0]
        # prediksiSR = modelSR.predict([[float(i) for i in request.GET['sr'].split(',')]])[0]

        # pressure
        if prediksiP > 1014:
            context['prediksiP'] = "Over"
        elif prediksiP > 1002:
            context['prediksiP'] = "Normal"
        else:
            context['prediksiP'] = "Warning"

        # temp
        if prediksiT > 37.92:
            context['prediksiT'] = "Over"
        elif prediksiT > 17.72:
            context['prediksiT'] = "Normal"
        else:
            context['prediksiT'] = "Warning"
        
        # RH
        if prediksiRH > 24.52:
            context['prediksiRH'] = "Normal"
        else:
            context['prediksiRH'] = "Warning"

        # wind speed
        if prediksiWS > 5.85:
            context['prediksiWS'] = "Over"
        elif prediksiWS > 3.6:
            context['prediksiWS'] = "Warning"
        else:
            context['prediksiWS'] = "Normal"

        # Wind Dir
        if prediksiWD > 17:
            context['prediksiWD'] = "Over"
        elif prediksiWD > 16.9:
            context['prediksiWD'] = "Warning"
        else:
            context['prediksiWD'] = "Normal"

        # Rain
        if prediksiRN > 10:
            context['prediksiRN'] = "Over"
        elif prediksiP > 9.5:
            context['prediksiRN'] = "Warning"
        else:
            context['prediksiRN'] = "Normal"

        # Water Level
        if prediksiWL > 4:
            context['prediksiWL'] = "Warning"
        else:
            context['prediksiWL'] = "Normal"
        
        # Water TEmp
        if prediksiWT > 35:
            context['prediksiWT'] = "Warning"
        else:
            context['prediksiWT'] = "Normal"

        # # Solar Rad
        # if prediksiSR > 590:
        #     context['prediksiSR'] = "Over"
        # elif prediksiP > 588.4:
        #     context['prediksiSR'] = "Warning"
        # else:
        #     context['prediksiSR'] = "Normal"

    return JsonResponse(context)


def status(request):
    context = {
        'konten': 'snippets/statusKonten.html',
    }
    return render(request, 'index.html', context)


def info(request):
    context = {
        'konten': 'snippets/info.html',
    }
    return render(request, 'index.html', context)
