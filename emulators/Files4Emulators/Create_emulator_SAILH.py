"""
Created on Feb. 26th  2022
@author: Jingwen Wang
------------------------------------------------------------------------------
This file creates the emulator for SAILH
------------------------------------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import shelve
import time
import RTMs

def fDir(wvl):
    fdir=RTMs.skylspec(wvl,0.19).reshape((-1,1))
    return fdir

def SoilRef(rsl1,rsl2,rsl3, rsl4,coef):
    out_ref = np.dot(coef, np.asarray((rsl1,rsl2,rsl3,rsl4)))
    return np.transpose(out_ref)

def LeafOptfromPROSPECT(N,Cab,Car,Cm,Cw,coef,wvl_in):
    out_ref=np.zeros((len(N),2101))
    out_tra=np.zeros((len(N),2101))
    for i in np.arange(0, len(N)):
        wvl, out_ref[i,:],out_tra[i,:] = RTMs.PROSPECT_5(N[i],Cab[i],Car[i],Cw[i],Cm[i],coef)
        if np.mod(i, 100) == 0:
            print('PROSPECT simulation ' + str(i) + ' out of ' + str(len(N)) + ' at '+ time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    return out_ref[:,np.asarray(wvl_in-400,dtype =int)],out_tra[:,np.asarray(wvl_in-400,dtype =int)]
def CanopyfromSAILH(LAI, ALA, hotspot, leafr, leaft, soilr, fdir, sun_theta, view_theta, phi_sunview):
    out_ref = np.zeros_like(leafr)
    for i in np.arange(0, len(LAI)):
        for j in np.arange(0, np.shape(leafr)[1]):
            out_ref[i, j] = RTMs.SAILHEllip(LAI[i], ALA[i], sun_theta[i], view_theta[i], phi_sunview[i], hotspot[i],
                                           leafr[i, j], leaft[i, j], soilr[i, j], fdir[j])
        if np.mod(i, 100) == 0:
            print('SAILH simulation' + str(i) + 'out of ' + str(len(LAI)) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return out_ref
#
Pr_coeff=RTMs.getPROSPECTcoef()

myshelf = shelve.open('BandSampling')
wvl_emu = myshelf['selected_Bands']
myshelf.close()
Pr_coeff=RTMs.getPROSPECTcoef()
SoilRefCoefEmu = RTMs.getSoilSpeccoef(wvl_emu)
fDiremu = fDir(wvl_emu)

####-------------------------------------------
#
nsims = 1000000
#================= 1. generate training/validation dataset======================#
#### --- input for prospect emulator
#
N=np.random.uniform(1,3,nsims)
Cab=np.random.uniform(0,100,nsims)
Car=np.random.uniform(0,25,nsims)
Cm=np.random.uniform(0,0.02,nsims)
Cw=np.random.uniform(0,0.02,nsims)
#### --- input for sailh emulator
LAI = np.random.uniform(0,10,nsims)
ALA = np.random.uniform(0,85,nsims)
hotspot = np.random.uniform(0.0001,0.4,nsims)
theta_sun=np.random.uniform(0,45,nsims)
theta_obs=np.random.uniform(0,30,nsims)
phi_sun_obs=np.random.uniform(0,180,nsims)
rsl1 = np.random.uniform(0.01,0.45,nsims)
rsl2 = np.random.uniform(-0.1,0.1,nsims)
rsl3 = np.random.uniform(-0.05,0.05,nsims)
rsl4 = np.random.uniform(-0.04,0.04,nsims)

emubandsoilr = SoilRef(rsl1,rsl2,rsl3,rsl4, SoilRefCoefEmu)
emubandleafr,emubandleaft = LeafOptfromPROSPECT(N,Cab, Car,Cm, Cw,Pr_coeff,wvl_emu)
emubandcanopy_ref = CanopyfromSAILH(LAI,ALA,hotspot,emubandleafr,emubandleaft,emubandsoilr,fDiremu,theta_sun,theta_obs,phi_sun_obs)

nrows = len(LAI)
nwvl = np.shape(emubandleafr)[1]
non_spectral=np.repeat(np.hstack((np.reshape(LAI,(-1,1)),np.reshape(ALA,(-1,1)),np.reshape(theta_sun,(-1,1)),
                                      np.reshape(theta_obs,(-1,1)),np.reshape(phi_sun_obs,(-1,1)),np.reshape(hotspot,(-1,1)))),nwvl,axis=0)
fdir_all = np.repeat(np.reshape(fDiremu,(1,-1)),nrows,axis=0)

input_matrix = np.hstack((non_spectral,emubandleafr.flatten().reshape(-1,1),emubandleaft.flatten().reshape(-1,1),emubandsoilr.flatten().reshape(-1,1),fdir_all.reshape(-1,1)))
y = emubandcanopy_ref.reshape(-1,1)
# if save the traning set for sailh
# myshelf = shelve.open("Trainingset_SAILH_emubands")
# myshelf['X'] = input_matrix
# myshelf['y'] = y
# myshelf.close()

SC_X = StandardScaler().fit(input_matrix)
SC_y = StandardScaler().fit(y)

X_train, X_test, y_train, y_test = train_test_split(input_matrix, y, test_size=0.3, random_state=0)
X_train_norm = SC_X.transform(X_train)
y_train_norm = SC_y.transform(y_train).flatten()
X_test_norm = SC_X.transform(X_test)
print("Start training ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
NN=MLPRegressor((85,45,50), activation='tanh',solver='adam',max_iter=6000,tol=0.00000001)
NN.fit(X_train_norm,y_train_norm)
print("Finished training ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Start predicting ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
y_pred_n =NN.predict(X_test_norm)
y_pred = SC_y.inverse_transform(y_pred_n).reshape(-1,1)
print("Finished predicting ref at ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
rmse = np.sqrt(np.sum(np.power(y_test - y_pred,2).flatten())/(y_pred.shape[0]*y_pred.shape[1]))
print('validation ref RMSE:%.5f' % rmse)

ds = shelve.open("emulator_SAILH_test")
ds['scaler_input']= SC_X
ds['scaler_output'] = SC_y
ds["NN"] = NN
ds.close()

plt.scatter(y_test,y_pred)
plt.text(0.05, 0.85, "RMSE is : %.5f" % rmse)
plt.show()
#