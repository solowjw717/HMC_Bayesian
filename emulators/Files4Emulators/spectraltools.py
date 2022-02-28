
def spectral_filter (in_wavelength,in_optical, spectralresponse):
    #in_wavelength : input wavelengths (nm)
    #in_optical : input optical properties (reflectance, transmittance) value
    #sensor : string, name of the sensor (in capitals)
    import numpy as np
    band_count=0
    out_optical=np.zeros(len(spectralresponse.names))
    for band_name in spectralresponse.names:
        relative_response=eval('spectralresponse.'+str(band_name)+'_RSR[0]')
        sensor_wavelength=eval('spectralresponse.'+str(band_name)+'_WVL[0]')
        in_optical_0=np.interp(sensor_wavelength,in_wavelength,in_optical)
        out_optical[band_count]=np.sum(in_optical_0*relative_response)/np.sum(relative_response)
        band_count=band_count+1
    return spectralresponse.names, out_optical

class spectralresponse():
    def __init__(self,sensor_name):
        import tables
        import numpy as np
        name_filters_file=r'D:\2020_P4_Emulator\Bayesian\Smac_filters_2015.h5' #File with sensor coefficients
        filters_file=tables.open_file(name_filters_file,mode='r',root_uep='/'+sensor_name)
        self.names=[]
        band_count=0
        for group in filters_file.walk_groups("/"):
            if group._v_depth==2:
                self.names.append(group._v_name)
                spec_res=np.array(group._f_get_child('rsr'))
                wvl=np.array(group._f_get_child('lambda'))
                exec('self.'+group._v_name+'_RSR=spec_res')
                if np.min(wvl)<1:
                    fact=1000
                else:
                    fact=1
                exec('self.'+group._v_name+'_WVL=wvl*fact')
        filters_file.close()
