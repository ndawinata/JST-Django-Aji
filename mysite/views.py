from django.shortcuts import render
import os
import pickle
import numpy as np
import requests
from datetime import datetime
from django.http import HttpResponse
from django.http import JsonResponse
import json

def index(request):
    context={
        'konten':'snippets/homeKonten.html',
        'hasil':"",
        'water':'none',
        'dateNow':'none',
        'hasilRealtime':'none',
    }
    if request.method == 'POST':
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, 'model.pkl')
        f = open(file_path, 'rb')
        model = pickle.load(f)
        f.close()
        context['level'] = request.POST['level']
        context['bulan'] = request.POST['bulan']
        y_pred = model.predict(np.array([[request.POST['level'],request.POST['bulan']]]))
        if y_pred < 0.5:
            context['hasil'] = "Tidak Berpotensi Banjir ROB"
        else:
            context['hasil'] = "Berpotensi Banjir ROB"

    return render(request, './index.html', context)

def jst(request):
    context={
        'konten':'snippets/jstKonten.html',
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
        y_pred = model.predict(np.array([[request.GET['level'],request.GET['bulan']]]))
        if y_pred < 0.5:
            context['hasilreal'] = "Tidak Berpotensi Banjir ROB"
            context['level'] = request.GET['level']
            context['bulan'] = request.GET['bulan']
        else:
            context['hasilreal'] = "Berpotensi Banjir ROB"
            context['level'] = request.GET['level']
            context['bulan'] = request.GET['bulan']
    return JsonResponse(context)

def banjir(request):
    context={
        'konten':'snippets/banjirKonten.html',
    }
    return render(request, 'index.html', context)