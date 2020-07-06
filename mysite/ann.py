import pandas as pd
import numpy as np
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
import io
import sys
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import ftplib
from ftplib import FTP
import pickle

data = pd.read_csv("datawaterlevel.csv",sep=",")
#data

# PENGUBAHAN FORMAT DATASET AGAR SESUAI DENGAN MODEL ANN KERAS
x = data.iloc[:,:2].values
y = data.iloc[:,2].values

# MELIHAT FORMAT INPUT YANG SESUAI DENGAN MODEL 
#x
#y

# INISIASI ANN (Artificial Neural Network)
model = Sequential()

# INISIASI INPUT LAYER SEBANYAK 2 LAYER DAN HIDDEN LAYER SEBANYAK 15 NODE
model.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# PENDAMBAHAN HIDDEN LAYER KEDUA 
model.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# PENDAMBAHAN HIDDEN LAYER KETIGA
model.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# PENDAMBAHAN HIDDEN LAYER KELIMA
model.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# PENAMBAHAN OUTPUT LAYER
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# COMPILE ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# TRAINING
hist = model.fit(x, y, batch_size = 24, epochs = 2000)

# CARA SAVE MODEL
model.save("model.h5")

# CARA MENGAMBIL MODEL
from keras.models import load_model

model = load_model('model.h5')
pickle.dump(model, open('model.pkl','wb'))
# PREDIKSI/INFERENCE

# # memprediksikan jika ada inputan pasut = 3 & bulan = 8 maka model akan memprediksikan spesies menjadi 1 atau 0
# a = np.array([[2.1,1]])
# y_pred = model.predict(a)

# # HASIL PREDIKSI
# print(y_pred)

# if(y_pred < 0.5):
#   print("TIDAK BERPOTENSI BANJIR ROB")
# else:
#   print("POTENSI BANJIR ROB")

# plt.plot(hist.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()
# plt.savefig('a.png')

# plt.plot(hist.history['accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()
# plt.savefig('b.png')

# ftp = FTP('files.000webhost.com')  
# ftp.login('websitejst', 'uXHlmXbS5XgkUEOAhtgA')  
# with open('a.png', 'rb') as f:  
#     ftp.storlines('STOR %s' % 'public_html/a.png', f)  
# ftp.quit()

# ftp = FTP('files.000webhost.com')  
# ftp.login('websitejst', 'uXHlmXbS5XgkUEOAhtgA')  
# with open('b.png', 'rb') as f:  
#     ftp.storlines('STOR %s' % 'public_html/b.png', f)  
# ftp.quit()