"""
Created on Feb. 26th  2022
@author: Jingwen Wang
------------------------------------------------------------------------------
This file creates the sensor-specfic emulator for band convolution (from the 40 wavelengths to sensor bands)
------------------------------------------------------------------------------
"""
import shelve
import spectraltools as st
import numpy as np
from scipy import sparse
from sklearn.model_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import *
import time
import RTMs

def fDir(wvl):
    fdir=RTMs.skylspec(wvl,0.19).reshape((-1,1))
    return fdir
def SoilRef(rsl1,rsl2,rsl3, rsl4,coef):
    out_ref = np.dot(coef, np.asarray((rsl1,rsl2,rsl3,rsl4)))
    return np.transpose(out_ref)

def CreatFullwvlCanopyset(nsims):
    ## creat training set####
    print('Creating training fullwavelength Canopy reflectance' + ' at '+ time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    wvl_full = np.arange(400, 2501)
    N = np.random.uniform(1, 3, nsims)
    Cab = np.random.uniform(0, 100, nsims)
    Car = np.random.uniform(0, 25, nsims)
    Cm = np.random.uniform(0, 0.02, nsims)
    Cw = np.random.uniform(0, 0.02, nsims)

    LAI = np.random.uniform(0, 10, nsims)
    ALA = np.random.uniform(0, 85, nsims)
    hotspot = np.random.uniform(0.0001, 0.4, nsims)
    theta_sun = np.random.uniform(0, 45, nsims)
    theta_obs = np.random.uniform(0, 30, nsims)
    phi_sun_obs = np.random.uniform(0, 180, nsims)
    rsl1 = np.random.uniform(0.01, 0.45, nsims)
    rsl2 = np.random.uniform(-0.1, 0.1, nsims)
    rsl3 = np.random.uniform(-0.05, 0.05, nsims)
    rsl4 = np.random.uniform(-0.04, 0.04, nsims)

    Pr_coeff = RTMs.getPROSPECTcoef()
    SoilRefCoef = RTMs.getSoilSpeccoef(wvl_full)
    fdir = fDir(wvl_full)

    # full wavelength training canopy reflectance
    fullleafr, fullleaft = LeafOptfromPROSPECT(N, Cab, Car, Cm, Cw, Pr_coeff)
    fullsoilr = SoilRef(rsl1, rsl2, rsl3, rsl4, SoilRefCoef)
    fullcanopy_ref = CanopyfromSAILH(LAI, ALA, hotspot, fullleafr, fullleaft, fullsoilr, fdir, theta_sun, theta_obs,
                                     phi_sun_obs)
    # if save
    myshelf = shelve.open("Canopy_ref_set_test")
    myshelf['Canopy_ref'] = fullcanopy_ref
    myshelf.close()

    return fullcanopy_ref

def CanopyfromSAILH(LAI,ALA,hotspot,leafr,leaft,soilr,fdir,sun_theta,view_theta,phi_sunview):
    out_ref=np.zeros_like(leafr)
    for i in np.arange(0,len(LAI)):
        for j in np.arange(0,np.shape(leafr)[1]):
            out_ref[i, j] = RTMs.SAILHEllip(LAI[i], ALA[i], sun_theta[i], view_theta[i], phi_sunview[i], hotspot[i],
                                            leafr[i, j], leaft[i, j], soilr[i, j], fdir[j])
            # print('SAILH simulation at wvl {0} in iteration {1}/{2} : time {3}'.format(j,i,len(LAI),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
        print('SAILH simulation ' + str(i) + ' out of ' + str(len(LAI)))
    return out_ref

def LeafOptfromPROSPECT(N,Cab,Car,Cm,Cw,coef):
    out_ref=np.zeros((len(N),2101))
    out_tra=np.zeros((len(N),2101))
    for i in np.arange(0, len(N)):
        wvl, out_ref[i,:],out_tra[i,:] = RTMs.PROSPECT_5(N[i],Cab[i],Car[i],Cw[i],Cm[i],coef)
        print('PROSEPCT simulation ' + str(i) + ' out of ' + str(len(N)) )
    return out_ref,out_tra

# emulating wavelength
myshelf = shelve.open('BandSampling')
wvl_emu = myshelf['selected_Bands']
myshelf.close()
SoilRefCoefEmu = RTMs.getSoilSpeccoef(wvl_emu)
fDiremu = fDir(wvl_emu)

##################################################################################
# Now train NNs for transfering the 40 wavelength ref into band ref
#
# read the pre-produced fullwavelength canopy reflectance set for training
# canopy_ref_set = shelve.open("Canopy_ref_set_test")['Canopy_ref']
# Or create your fullwavelength canopy reflectance set for training
canopy_ref_set = CreatFullwvlCanopyset(nsims = 20000)

X = canopy_ref_set[:, wvl_emu.astype(int) - 400] # reflectance at the 40 emulation wavelengths

NN_ref = []
NN_ref_Scaler_input = []
NN_ref_Scaler_output = []
sensor_names = ['QUICKBIRD','LANDSAT8','S2A']
sensor_band_names = {'QUICKBIRD':['B1','B2','B3','B4'],'LANDSAT8':['B2','B3','B4','B5','B6','B7'],'S2A':['B2','B3','B4','B5','B6','B7','B8','B8a','B11','B12'],'S2B':['B2','B3','B4','B5','B6','B7','B8','B8a','B11','B12']}
sensor_nbands ={'QUICKBIRD':4,'LANDSAT8':6,'S2A':10}
NN_ref = shelve.open("emulator_BC_test")

for sensor in sensor_names:
    ##### Aggregate wvl to band reflectance (creat y)
    sensorRSR = st.spectralresponse(sensor)
    WVL = []
    RSR = []
    for band in sensor_band_names[sensor]:
        rsr_abs = 0
        WVL.append(eval('sensorRSR.'+ band+ '_WVL[0]'))
        rsr_abs = eval('sensorRSR.' + band + '_RSR[0]')
        rsr_rel = rsr_abs / np.sum(rsr_abs)
        RSR.append(rsr_rel)
    # wavelengths-to-band aggregation matrices
    map_matrix = np.zeros((len(sensor_band_names[sensor]), 2101))
    for band in np.arange(0, len(sensor_band_names[sensor])):
        c_wvl = WVL[band]
        c_rsr = RSR[band]
        map_matrix[band, (c_wvl - 400).astype(int).tolist()] = c_rsr
    wvl_to_bands_mapping = sparse.csr_matrix(map_matrix)
    reflectance_bands = np.asarray(
        np.dot(wvl_to_bands_mapping.todense(), canopy_ref_set.transpose())).transpose()  ##########New

    ##### Start training the band convolution NN for this sensor
    X_train, X_test, y_train, y_test = train_test_split(X, reflectance_bands, test_size=0.3, random_state=0)
    SC_X = StandardScaler().fit(X)
    SC_y = StandardScaler().fit(reflectance_bands)
    NN_ref[sensor + "_SC_X"] = SC_X
    NN_ref[sensor + "_SC_y"] = SC_y
    print("========  Start training bc NN {} at {}".format(sensor, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                 time.localtime(time.time()))))
    NN = MLPRegressor((10, 10), activation='relu', solver='lbfgs', max_iter=6000, tol=0.00000001)
    NN.fit(SC_X.transform(X_train + np.random.normal(0, 0.0001, (np.shape(X_train)))), SC_y.transform(y_train))
    print("Finish training bcNN {} at {}".format(sensor, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                       time.localtime(time.time()))))
    print("Start predicting bcNN {} at {}".format(sensor, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                        time.localtime(time.time()))))
    y_pred = SC_y.inverse_transform(NN.predict(SC_X.transform(X_test)))
    print("Finished predicting bcNN {} at {}".format(sensor, time.strftime('%Y-%m-%d %H:%M:%S',
                                                                           time.localtime(time.time()))))
    NN_ref[sensor + "_NN"] = NN
    rmse = np.sqrt(np.sum(np.power(y_test - y_pred, 2).flatten()) / (y_pred.shape[0] * y_pred.shape[1]))
    print('Validation bcNN RMSE:%.5f' % rmse)

NN_ref.close()


