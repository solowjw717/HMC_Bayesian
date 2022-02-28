"""
Created on Feb. 26th  2022
@author: Jingwen Wang
------------------------------------------------------------------------------
This file creates the emulator for PROSPECT-5
------------------------------------------------------------------------------
"""

import numpy as np
from sklearn.model_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import shelve
import time
import RTMs
#
Pr_coeff=RTMs.getPROSPECTcoef()
# read the selected 40 bands for emulation
file = shelve.open("BandSampling")
wvl_emu = file["selected_Bands"]
file.close()

sampleindex = (wvl_emu - 400).astype(int)
Pr_coeff_sample = Pr_coeff[sampleindex]

#================= 1. generate training/validation dataset======================#
nsims = 1000000

N=np.random.uniform(1,3,nsims)
Cab=np.random.uniform(0,100,nsims)
Car=np.random.uniform(0,25,nsims)
Cm=np.random.uniform(0,0.02,nsims)
Cw=np.random.uniform(0,0.02,nsims)

l_ref = np.zeros((nsims, len(wvl_emu)))
l_tra = np.zeros((nsims, len(wvl_emu)))

for i in range(0,nsims):
    wvl, l_ref[i], l_tra[i] = RTMs.PROSPECT_5(N[i],Cab[i],Car[i],Cw[i],Cm[i], Pr_coeff_sample)
    print('PROSPECT simulation ' + str(i) + 'out of ' + str(nsims))


X = np.hstack((N.reshape(-1,1),Cab.reshape(-1,1),Car.reshape(-1,1),Cw.reshape(-1,1),Cm.reshape(-1,1)))
y = np.hstack((l_ref,l_tra))
SC_X = StandardScaler().fit(X)
SC_y = StandardScaler().fit(y)

#==================================== 2. start training==================================#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = y_train + np.random.normal(0, 0.000001) # A random error is added to overcome possible overfitting
X_train_norm = SC_X.transform(X_train)
y_train_norm = SC_y.transform(y_train)

print("Start training ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
NN=MLPRegressor((30,40), activation='tanh',solver='adam',max_iter=6000,tol=0.00000001)
NN.fit(X_train_norm,y_train_norm)
print("Finished training ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

print("Start predicting ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
y_pred = SC_y.inverse_transform(NN.predict(SC_X.transform(X_test)))
print("Finished predicting ref&tra at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
rmse= np.sqrt(np.sum(np.power(y_test - y_pred,2).flatten())/(y_pred.shape[0]*y_pred.shape[1]))
print('validation ref RMSE:%.5f' % rmse)


ds = shelve.open("emulator_PROSPECT_test")
ds['scaler_input']= SC_X
ds['scaler_output'] = SC_y
ds["NN"] = NN
ds["selected_Bands"] = wvl_emu
ds.close()
plt.scatter(y_test,y_pred)
plt.text(0.05, 0.55, "RMSE is : %.5f" % rmse)
plt.show()
#