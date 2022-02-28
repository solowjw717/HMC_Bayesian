"""
Created on Feb. 26th  2022
@author: Jingwen Wang, Raul Lopez-Lozano
---------------------------------------------------------------------------------------------------------------------------
This file tests the performance of created emulators, with different cases seperating the errors introduced by different emulation steps
---------------------------------------------------------------------------------------------------------------------------
* The difference of this script from test_emulator is that this implementation doesn't take wavelength effects of soil_ref and fdir into consideration
"""
import shelve
import numpy as np
import spectraltools as st
import RTMs as RTM
from sklearn.model_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import *
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interpn

def bandsReffromNN(reflectance,ScalerIn,ScalerOut, NN):
    out_ref_bands = (ScalerOut.inverse_transform(NN.predict(ScalerIn.transform(reflectance))))
    return out_ref_bands

def createMatrixRSR(sensor,bands):
    sensorRSR = st.spectralresponse(sensor)
    b = 0
    first = True
    for band in bands:
        if first:
            WVL=eval('sensorRSR.' + band + '_WVL[0]')
            rsr_abs = eval('sensorRSR.' + band + '_RSR[0]')
            rsr_rel = rsr_abs / np.sum(rsr_abs)
            RSR = (rsr_rel)
            band_id = np.ones_like(rsr_rel) * b
            first=False
        else:
            WVL=np.hstack((WVL,eval('sensorRSR.' + band + '_WVL[0]')))
            rsr_abs = eval('sensorRSR.' + band + '_RSR[0]')
            rsr_rel = rsr_abs / np.sum(rsr_abs)
            RSR= np.hstack((RSR,rsr_rel))
            band_id = np.hstack((band_id, np.ones_like(rsr_rel) * b))
        b += 1
    wvl_to_simulate = np.unique(WVL)
    # wavelengths-to-band aggregation matrices
    map_matrix = np.zeros((len(bands),len(wvl_to_simulate)))
    for band in np.arange(0, len(bands)):
        pos_band = band_id == band
        c_wvl = WVL[pos_band]
        c_rsr = RSR[pos_band]
        for i in np.arange(0, len(c_wvl)):
            map_matrix[band, wvl_to_simulate == c_wvl[i]] = c_rsr[i]
    wvl_to_bands_mapping = sparse.csr_matrix(map_matrix)
    return wvl_to_bands_mapping, wvl_to_simulate

def bandsReffromRSR(reflectance, matrix):
    out_ref_bands = np.asarray(np.dot(matrix.todense(), reflectance.transpose())).transpose()
    return out_ref_bands

def LeafOptfromPROSPECT(N,Cab,Car,Cw,Cm,coef,wvl_in):
    out_ref=np.zeros((len(N),2101))
    out_tra=np.zeros((len(N),2101))
    for i in np.arange(0, len(N)):
        wvl, out_ref[i,:],out_tra[i,:] = RTM.PROSPECT_5(N[i],Cab[i],Car[i],Cw[i],Cm[i],coef)
    return out_ref[:,np.asarray(wvl_in-400,dtype =int)],out_tra[:,np.asarray(wvl_in-400,dtype =int)]
    # return out_ref,out_tra

def LeafOptfromEmu(N,Cab,Car,Cw,Cm):
    # myshelf = shelve.open('Trainingset_PROSPECT_40')
    # print(list(myshelf))
    # NN = myshelf["NN_PROSPECT"]
    # ScalerIn = myshelf['scale_X']
    # ScalerOut = myshelf['scale_y']
    myshelf = shelve.open('emulator_PROSPECT_test')
    NN = myshelf["NN"]
    ScalerIn = myshelf['scaler_input']
    ScalerOut = myshelf['scaler_output']
    myshelf.close()
    parameters = np.hstack((N.reshape(-1,1),Cab.reshape(-1,1),Car.reshape(-1,1),Cw.reshape(-1,1),Cm.reshape(-1,1)))
    l_ref_tra = ScalerOut.inverse_transform((NN.predict(ScalerIn.transform(parameters))))
    out_ref = l_ref_tra[:, 0: 40]
    out_tra = l_ref_tra[:, 40:]
    return out_ref, out_tra

def SoilRef(rsl1,rsl2,rsl3, rsl4,coef):
    out_ref = np.dot(coef, np.asarray((rsl1,rsl2,rsl3,rsl4)))
    return np.transpose(out_ref)

def CanopyfromSAILH(LAI,ALA,hotspot,leafr,leaft,soilr,fdir,sun_theta,view_theta,phi_sunview):
    out_ref=np.zeros_like(leafr)
    for i in np.arange(0,len(LAI)):
        for j in np.arange(0,np.shape(leafr)[1]):
            out_ref[i,j] = RTM.SAILHEllip(LAI[i],ALA[i],sun_theta[i],view_theta[i],phi_sunview[i],hotspot[i],leafr[i,j],leaft[i,j],soilr[i],fdir[i])
    return out_ref

def CanopyfromEmu(LAI,ALA,hotspot,leafr,leaft,soilr,fdir,sun_theta,view_theta,phi_sunview):
    myshelf = shelve.open('emulator_SAILH_test')
    NN = myshelf['NN']
    ScalerIn = myshelf['scaler_input']
    ScalerOut = myshelf['scaler_output']
    myshelf.close()
    nrows = len(LAI)
    nwvl = np.shape(leafr)[1]
    non_spectral=np.repeat(np.hstack((np.reshape(LAI,(-1,1)),np.reshape(ALA,(-1,1)),np.reshape(sun_theta,(-1,1)),
                                      np.reshape(view_theta,(-1,1)),np.reshape(phi_sunview,(-1,1)),np.reshape(hotspot,(-1,1)))),nwvl,axis=0)
    # fdir_all = np.repeat(np.reshape(fdir,(-1,1)),nrows,axis=0)
    input_matrix=np.hstack((non_spectral,leafr.flatten().reshape(-1,1),leaft.flatten().reshape(-1,1),np.repeat(soilr.reshape(-1,1),nwvl,axis=0),np.repeat(fdir.reshape(-1,1),nwvl,axis=0)))
    out_ref = (ScalerOut.inverse_transform((NN.predict(ScalerIn.transform(input_matrix))).reshape(-1, 1))).flatten()
    return np.reshape(out_ref,(nrows,nwvl))

def fDir(wvl):
    fdir=RTM.skylspec(wvl,0.19).reshape((-1,1))
    return fdir

def density_scatter( x , y, ax, bins = 10, i= 0,rmse=0, rrmse=0, **kwargs )   :
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn(( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    im = ax.scatter( x, y, c=z,cmap = plt.cm.rainbow, **kwargs)
    ax.text(0.06,ax.get_ylim()[1]*0.85,"case"+str(i+1))
    ax.text(0.2,0.04, "RMSE=" + format(rmse, '.4f') + "(" + format(rrmse, '.2f') + "%)")
    return im,z


sensors = ['QUICKBIRD','LANDSAT8','S2A']
sensor_band_names = {'QUICKBIRD':['B1','B2','B3','B4'],'LANDSAT8':['B2','B3','B4','B5','B6','B7'],'S2A':['B2','B3','B4','B5','B6','B7','B8','B8a','B11','B12'],'S2B':['B2','B3','B4','B5','B6','B7','B8','B8a','B11','B12']}

for sensor in sensors:
    bands = sensor_band_names[sensor]
    file = shelve.open("BandSampling")
    wvl_emu = file["selected_Bands"]
    file.close()
    #
    #
    myshelf = shelve.open("emulator_BC_test")
    NN = myshelf[sensor + "_NN"]
    NNScaler_in = myshelf[sensor + "_SC_X"]
    NNScaler_out = myshelf[sensor + "_SC_y"]
    myshelf.close()
    #
    matrix_aggregation, wvl_to_simulate = createMatrixRSR(sensor,bands)
    PROSPECTCoef=RTM.getPROSPECTcoef()

    nsims=1000
    LAI = np.random.uniform(0,10,nsims)
    ALA = np.random.uniform(0,85,nsims)
    hotspot = np.random.uniform(0,0.4,nsims)
    theta_sun=np.random.uniform(0,45,nsims)
    theta_obs=np.random.uniform(0,10,nsims)
    phi_sun_obs=np.random.uniform(0,180,nsims)
    N=np.random.uniform(1,3,nsims)
    Cab=np.random.uniform(0,100,nsims)
    Car=np.random.uniform(0,25,nsims)
    Cm=np.random.uniform(0,0.02,nsims)
    Cw=np.random.uniform(0,0.02,nsims)
    soilr = np.random.uniform(0, 0.4, nsims)
    fDir = np.random.uniform(0.6, 1.0, nsims)

    #CASE 0 full models
    fullleafr,fullleaft = LeafOptfromPROSPECT(N,Cab, Car,Cw, Cm, PROSPECTCoef,wvl_to_simulate)
    # fullsoilr = SoilRef(rsl1,rsl2,rsl3,rsl4, SoilRefCoefall)
    fullcanopy_ref = CanopyfromSAILH(LAI,ALA,hotspot,fullleafr,fullleaft,soilr,fDir,theta_sun,theta_obs,phi_sun_obs)
    fullcanopy_bands=bandsReffromRSR(fullcanopy_ref,matrix_aggregation)

    #CASE 1 original PROSPECT and SAILH but only on the 40 bands, then aggregation from NN
    emubandleafr,emubandleaft = LeafOptfromPROSPECT(N,Cab, Car,Cw, Cm, PROSPECTCoef,wvl_emu)
    # emubandsoilr = SoilRef(rsl1,rsl2,rsl3,rsl4, SoilRefCoefEmu)
    emubandcanopy_ref = CanopyfromSAILH(LAI,ALA,hotspot,emubandleafr,emubandleaft,soilr,fDir,theta_sun,theta_obs,phi_sun_obs)
    emubandcanopy_bands=bandsReffromNN(emubandcanopy_ref,NNScaler_in,NNScaler_out,NN)

    #CASE 2 original PROSPECT on 40 bands, then emulator for SAILH and aggregation from NN
    emubandSAILHcanopy_ref = CanopyfromEmu(LAI,ALA,hotspot,emubandleafr,emubandleaft,soilr,fDir,theta_sun,theta_obs,phi_sun_obs)
    emubandcanopySAILH_bands=bandsReffromNN(emubandSAILHcanopy_ref,NNScaler_in,NNScaler_out,NN)

    #CASE 3 all steps are emulated
    emuleafr,emuleaft = LeafOptfromEmu(N,Cab, Car,Cw, Cm)
    emucanopy_ref =  CanopyfromEmu(LAI,ALA,hotspot,emuleafr,emuleaft,soilr,fDir,theta_sun,theta_obs,phi_sun_obs)
    emucanopy_bands=bandsReffromNN(emucanopy_ref,NNScaler_in,NNScaler_out,NN)

    rmse_bandsemu=np.sqrt(np.mean(np.power(fullcanopy_bands.flatten()-emubandcanopy_bands.flatten(),2)))
    rmse_bandsSAILHemu=np.sqrt(np.mean(np.power(fullcanopy_bands.flatten()-emubandcanopySAILH_bands.flatten(),2)))
    rmse_fullsemu=np.sqrt(np.mean(np.power(fullcanopy_bands.flatten()-emucanopy_bands.flatten(),2)))

    figure,axes=plt.subplots(1,3,figsize = (10,3),sharey=True)
    im,z = density_scatter(fullcanopy_bands.flatten(), emubandcanopy_bands.flatten(), ax=axes[0], i=0,rmse = rmse_bandsemu,rrmse=100*rmse_bandsemu/np.mean(fullcanopy_bands.flatten()))
    density_scatter(fullcanopy_bands.flatten(), emubandcanopySAILH_bands.flatten(), ax=axes[1], i=1,rmse = rmse_bandsSAILHemu,rrmse=100*rmse_bandsSAILHemu/np.mean(fullcanopy_bands.flatten()))
    density_scatter(fullcanopy_bands.flatten(), emucanopy_bands.flatten(), ax=axes[2], i=2,rmse = rmse_fullsemu,rrmse=100*rmse_fullsemu/np.mean(fullcanopy_bands.flatten()))
    figure.subplots_adjust(bottom=0.2, top=0.92, left=0.1, right=0.98,
                           wspace=0.05)
    cbar = figure.colorbar(im, ax=axes.ravel().tolist(), shrink=0.98)
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.ax.set_title("Density",fontsize =9 )
    figure.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(sensor+": Modelled Reflectance")
    plt.ylabel("Emulated Reflectance")

    # print('case 1 : ERROR from SAILH_emu is :%.5f' % rmse_SAILHemu)
    print('case 1 : ERROR from Band_Abstraction is :{}({})'.format(rmse_bandsemu,100*rmse_bandsemu/np.mean(fullcanopy_bands.flatten())))
    print('case 2 : ERROR from Band_Abstraction and SAILH_emu is :{}({})'.format(rmse_bandsSAILHemu,100*rmse_bandsSAILHemu/np.mean(fullcanopy_bands.flatten())))
    print('case 3 : ERROR from Band_Abstraction and PROSPECT_emu and SAILH_emu is :{}({})'.format(rmse_fullsemu,100*rmse_fullsemu/np.mean(fullcanopy_bands.flatten())))
    # figfile = r"D:\2020_P4_Emulator\Figure_MS\Emulator_"+sensor+".jpg"
    # figure.savefig(figfile)
    plt.show()