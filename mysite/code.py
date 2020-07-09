# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:53:10 2020

@author: I Nyoman P. Trisna
"""
# remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import joblib

test_size = 0.2

# create sliding windows
def sliding_window(data, window=5):
    window += 1
    data_window = []
    for i in range(len(data)-window+1):
        sub_data = []
        for j in range(window):
            sub_data.append(data[i+j])
        data_window.append(sub_data)
    df = pd.DataFrame(data_window)
    df.columns = ['x'+str(i) for i in range(window-1)]+['y']
    return df

data = pd.read_excel('Januari 2019.xlsx')
waktu = data['waktu']
data = data[['windspeed','winddir','temp','rh','pressure','rain','watertemp','waterlevel','solrad']]

for col in data.columns:
    data_windows = data[col].copy()
    data_windows = sliding_window(data_windows,10)
    
    data_x = data_windows.drop('y',axis=1)
    data_y = data_windows['y']
    x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                        data_y,
                                                        test_size=test_size,
                                                        shuffle=False,
                                                        random_state=123)
    
    svr_rbf = SVR(kernel='rbf')
    svr_lin = SVR(kernel='linear')
    
    filename = col
    
    data_result = pd.DataFrame()
    data_result['real'] = y_test
    
    if col == 'solrad':
        rmse_rbf = 'Undefined'
        mae_rbf = 'Undefined'
    else:
        regressor = svr_rbf
        regressor.fit(x_train,y_train)
        joblib.dump(regressor, filename+'_rbf.pkl')
        y_predict = regressor.predict(x_test)
        mae_rbf = np.square(mean_absolute_error(y_test,y_predict))
        rmse_rbf = np.square(mean_squared_error(y_test,y_predict))
        data_result['predicted (rbf)'] = y_predict   
    
    if col == 'solrad':
        regressor = LinearRegression()
    else:
        regressor = svr_lin
    regressor.fit(x_train,y_train)
    joblib.dump(regressor, filename+'_linear.pkl')
    y_predict = regressor.predict(x_test)
    mae_lin = np.square(mean_absolute_error(y_test,y_predict))
    rmse_lin = np.square(mean_squared_error(y_test,y_predict))
    data_result['predicted (linear)'] = y_predict
    
    for column in data_result.columns:
        data_result[column+" label"] = data_result[column]
    if col == "pressure":
        data_result['real label'] = ["Over" if i > 1010 else "Normal" if i > 1002 else "Warning" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Over" if i > 1010 else "Normal" if i > 1002 else "Warning" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Over" if i > 1010 else "Normal" if i > 1002 else "Warning" for i in data_result['predicted (linear) label']]
    elif col == "temp":
        data_result['real label'] = ["Over" if i > 37.92 else "Normal" if i > 17.72 else "Warning" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Over" if i > 37.92 else "Normal" if i > 17.72 else "Warning" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Over" if i > 37.92 else "Normal" if i > 17.72 else "Warning" for i in data_result['predicted (linear) label']]
    elif col == "rh":
        data_result['real label'] = ["Normal" if i > 24.52 else "Warning" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Normal" if i > 24.52 else "Warning" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Normal" if i > 24.52 else "Warning" for i in data_result['predicted (linear) label']]
    elif col == "windspeed":
        data_result['real label'] = ["Over" if i > 5.85 else "Warning" if i > 3.6 else "Normal" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Over" if i > 5.85 else "Warning" if i > 3.6 else "Normal" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Over" if i > 5.85 else "Warning" if i > 3.6 else "Normal" for i in data_result['predicted (linear) label']]
    elif col == "solrad":
        data_result['real label'] = ["Over" if i > 590 else "Warning" if i > 588.4 else "Normal" for i in data_result['real label']]
        data_result['predicted (linear) label'] = ["Over" if i > 590 else "Warning" if i > 588.4 else "Normal" for i in data_result['predicted (linear) label']]
    elif col == "waterlevel":
        data_result['real label'] = ["Warning" if i > 4 else "Normal" for i in data_result['real label']]
        data_result['predicted (linear) label'] = ["Warning" if i > 4 else "Normal" for i in data_result['predicted (linear) label']]
        data_result['predicted (rbf) label'] = ["Warning" if i > 4 else "Normal" for i in data_result['predicted (rbf) label']]
    elif col == "watertemp":
        data_result['real label'] = ["Warning" if i > 35 else "Normal" for i in data_result['real label']]
        data_result['predicted (linear) label'] = ["Warning" if i > 35 else "Normal" for i in data_result['predicted (linear) label']]
        data_result['predicted (rbf) label'] = ["Warning" if i > 35 else "Normal" for i in data_result['predicted (rbf) label']]
    elif col == "winddir":
        data_result['real label'] = ["Over" if i > 17 else "Warning" if i > 16.9 else "Normal" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Over" if i > 17 else "Warning" if i > 16.9 else "Normal" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Over" if i > 5.85 else "Warning" if i > 3.6 else "Normal" for i in data_result['predicted (linear) label']]
    elif col == "rain":
        data_result['real label'] = ["Over" if i > 10 else "Warning" if i > 9.5 else "Normal" for i in data_result['real label']]
        data_result['predicted (rbf) label'] = ["Over" if i > 10 else "Warning" if i > 9.5 else "Normal" for i in data_result['predicted (rbf) label']]
        data_result['predicted (linear) label'] = ["Over" if i > 5.85 else "Warning" if i > 3.6 else "Normal" for i in data_result['predicted (linear) label']]
    else:
        data_result = data_result.drop(data_result.columns[:len(data_result.columns)//2],axis=1)
    
    data_result.index = waktu.tail(len(data_result))
    data_result.to_csv('forecast_'+col+'.csv')
    
    print(col,rmse_rbf,mae_rbf,rmse_lin,mae_lin)