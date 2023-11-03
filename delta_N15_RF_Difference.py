from ncload import ncload
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
import math
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

'''##################################################################################
######################################################################################
######################        Random Forest algorithm   ############################
######################        for d15N_plant - d15N_soil   ############################         
Finished by Maoyuan Feng (fengmy@pku.edu.cn)
Contact: "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)"
'''
# subroutine for the calculation of earth area
def pxlarea(lati,latr,lonr):
    # lati = upper initial lat
    # latr = lat span angle
    # lonr = lon span angle
    # sphere surface area: S=2pi*r*r*(1-sin(theta))
    r = 6371.0 # unit: km
    pppi = math.pi
    alfa = (math.sin(lati/180.*math.pi)-math.sin((lati-latr)/180.*math.pi))
    beta = np.float64(lonr)/360.
    cal_area0 = 2*math.pi*r**2*alfa*beta
    cal_area = abs(2.*math.pi*r**2*(math.sin(np.float64(lati)/180.*math.pi)-math.sin((np.float64(lati)-np.float64(latr))/180.*math.pi))*np.float64(lonr)/360.)
    return cal_area

# Calculate the grid area
lat_area = np.arange(-90.0,90.00,0.1)
lon_area = np.arange(0,360.,0.1)
grid_area = np.zeros(len(lat_area))
for ii in range(0,len(lat_area)):
    grid_area[ii] = pxlarea(lat_area[ii],-0.1,0.1)  # unit: km2
grid_area[-1] = 0.0

# set the lontitude and latitude used in this code
lon = np.arange(-180,180,0.1)
lat = np.arange(-56,84,0.1)

###################################################################
########################   Data loading ###########################

# load the mask of desert grid cells
desert_file = ncload('Dersert_percent.nc')
desert_value = desert_file.get('new_data')

new_data = np.array(desert_value[:])
mask_desert = (new_data>=85) # mask out the grid cells when the percent of desert is >85%


# load the 16 predictors of RF model

global_data_path = '**************'

file_name_list = ['gpp1982_2015.nc','nhx_tian_2005_2014.nc','noy_tian_2005_2014.nc',\
                  'p_pet1980_2018.nc','pre1980_2018.nc','tmp1980_2018.nc',\
                  'BD.nc','CLAY.nc','SAND.nc','SILT.nc','OC.nc','TC.nc','TN.nc',\
                  'PHH2O.nc','am.nc','em.nc','nfix.nc']
# list of the variable names for the 16 predictors
var_name_list = ['gpp','nhx','noy','p_pet','pre','tmp',\
                 'BD','CLAY','SAND','SILT','OC','TC','TN','PHH2O',\
                 'am','em','nfix']

# load the .nc files, get the variable, and transform the matrices (lat*lon) into vects (nrow*1)
kk=0
for file_name in file_name_list:
    exec('file_p = ncload(global_data_path+"%s")'%file_name)
    exec('%s = file_p.get("%s")'%(var_name_list[kk],'tmp'))
    exec('%s_mat = np.array(%s[:].filled(np.nan))'%(var_name_list[kk],var_name_list[kk]))
    exec('%s_vect = np.squeeze(%s_mat.reshape(-1,1))'%(var_name_list[kk],var_name_list[kk]))
    kk = kk+1

# load the PFT file
pft_frac_nc = ncload('PFT_map_10th_deg.nc')
pft_frac = np.array(pft_frac_nc.get('pft_frac')[:].filled(np.nan))
pft_map_new = np.array(pft_frac[1:])
pft_map_new[9] = 0.0  # exclude the croplands and pastural regions
pft_map_new[11:14] = 0.0 # exclude the croplands and pastural regions

pft_sum = np.nansum(pft_map_new,axis=0) # the fraction of natural ecosystems (except desert)

# laod the land fraction file
land_frac_nc = ncload('land_fraction_from_CRU_10th.nc')
land_frac_data = land_frac_nc.get('land_frac')
land_frac_data1 = land_frac_data[:]*grid_area[:,np.newaxis]*1e6
land_frac_new = np.array(land_frac_data1[340:1740].filled(np.nan))*pft_sum[340:1740]

land_area_tot = np.array(land_frac_data1[340:1740].filled(np.nan))*np.nansum(pft_frac[1:,340:1740],axis=0)
land_area_tot_all = np.array(land_frac_data1[340:1740].filled(np.nan))*np.nansum(pft_frac[:,340:1740],axis=0)

land_frac_mat = np.transpose(np.flipud(land_frac_new),(1,0))
land_frac_vect = np.squeeze(land_frac_mat.reshape(-1,1))

# Prepare the predictors as vectors;
X_global = np.zeros((5040000,16)) # X_global is the matrix of the 16 predictors of RF model

mask_true = (BD[:].filled(np.nan) == BD[:].filled(np.nan))
mask_nan = np.logical_not(mask_true) # get the mask of the grids with NaN

X_global[:,0] = BD_vect[:]
X_global[:,1] = CLAY_vect[:]
X_global[:,2] = OC_vect[:]
X_global[:,3] = PHH2O_vect[:]
X_global[:,4] = SAND_vect[:]
X_global[:,5] = SILT_vect[:]
X_global[:,6] = am_vect[:]
X_global[:,7] = em_vect[:]
X_global[:,8] = gpp_vect[:]
X_global[:,9] = nfix_vect[:]
X_global[:,10] = nhx_vect[:]
X_global[:,11] = noy_vect[:]
X_global[:,12] = pre_vect[:]
X_global[:,13] = tmp_vect[:]
X_global[:,14] = p_pet_vect[:]
X_global[:,15] = np.where(TN_vect[:]<1e-3,100000,TC_vect[:]/TN_vect[:])

# get the mask of nan values
mask_nanana = (X_global!=X_global)
# get the mask of infinite large values
mask_inf = np.argwhere(np.isinf(X_global))
X_global[mask_nanana] = 0.5
# set the nan values to a given value 0.5;
# this value is only used in very limited grids, which are excluded in the final map
X_global[mask_inf] = 100000
# set the inf values to a large value 100000

X_global_pd = pd.DataFrame(X_global)
print(X_global_pd.isnull().any())
print(np.isnan(X_global).any())
print(np.isinf(X_global).any())
print(np.argwhere(np.isinf(X_global)))

# load the foliar d15N global map produced from RF model
RF_foliar_file = ncload('RF_foliar_N15.nc')
RF_N15_foliar = RF_foliar_file.get('N15_foliar')

# load the soil d15N global map produced from RF model
RF_soil_file = ncload('RF_soil_N15_Gaussian_0.05_m13_withRockSA_pasture_excluded_tian_C.nc')
RF_N15_soil = RF_soil_file.get('N15_soil')

# load the original leaf and soil d15N observations
Craine_foliar_data = '*****'
Craine_soil_data = '*****'

foliar_data = pd.read_excel(Craine_foliar_data,header=0,index_col=0)
soil_data = pd.read_excel(Craine_soil_data,header=0,index_col=0)

# drop the nan/empty values
foliar_new=foliar_data.dropna()
soil_new=soil_data.dropna()

# calculate the TC/TN ratio
foliar_rf_tc = foliar_new['TC'].values
foliar_rf_tn = foliar_new['TN'].values

soil_rf_tc = soil_new['TC'].values
soil_rf_tn = soil_new['TN'].values

foliar_new['C/N'] = np.where(foliar_rf_tn<1e-3,100000,foliar_rf_tc/foliar_rf_tn)
soil_new['C/N'] = np.where(soil_rf_tn<1e-3,100000,soil_rf_tc/soil_rf_tn)

# keep all vects of the 16 predictors, and drop all the useless value
foliar_rf = foliar_new.drop(labels=['Latitude','Longitude','AWC_CLASS','TC','TN'],axis=1)
soil_rf = soil_new.drop(labels=['Latitude','Longitude','AWC_CLASS','TC','TN'],axis=1)

# pick up the soil d15N values in the RF-based global map at the locations of foliar observations
# for the purpose of creating artificial pairwise observations of d15N-plant and d15N-soil
position = foliar_new[['Latitude','Longitude']]
Soil_extracted = np.zeros((len(position),))
for ii in range(0,len(position)):

    lat_value = position['Latitude'].iloc[ii]
    lon_value = position['Longitude'].iloc[ii]

    lat_pos = (lat_value+56)/0.1
    lon_pos = (lon_value+180)/0.1
    print(ii,lat_value,lon_value,lat_pos,lon_pos)
    Soil_extracted[ii] = RF_N15_soil[lon_pos,lat_pos]

foliar_mat = np.squeeze(np.array(foliar_list))
soil_mat = np.squeeze(np.array(soil_list))

# labels of the 16 predictors
feature_labels = pd.Series(['BD','Clay','OC','pH','Sand','Silt','AM','EM','GPP','Nfix','NHx','NOy','P','T','P/ET','C/N'])

# X1 is the predictors for training the RF model
X1 = (foliar_rf.drop(labels=['N15','pet'],axis=1)).values
#Y1 = Foliar_extracted[:]-(soil_rf['N15']).values
Y1 = (foliar_rf['N15']).values - Soil_extracted[:]
# Y1 is the d15N difference for training the RF model

# keep all true values
mask_y1_true = Y1==Y1
Y = Y1[mask_y1_true]
X = np.zeros((len(Y),16))

for ii in range(0,16):
    X[:,ii]=X1[mask_y1_true,ii]

# set up the Random Forest Model
forest = RandomForestRegressor(n_estimators=500,random_state=0,bootstrap=True,oob_score=True,min_samples_leaf=1,max_features='sqrt',n_jobs=4)
forest.fit(X,Y) # Train the RF model

# Predict the global estimates of d15N difference
N15_global = np.zeros((5040000,))
N15_global = forest.predict(X_global)
N15_global[mask_nanana[:,1]] = np.nan # mask out the grid cells with NaN values in predictors

diff_foliar_soil = np.transpose(np.fliplr(N15_global.reshape(3600,-1)),(1,0)) # Reshape the predicted vector into a map
diff_foliar_soil[mask_desert] = np.nan # mask out the desert grids

# Estimate the ensembles of d15N difference using the Trees in the RF model
N15_std_global_mat = np.zeros((5040000,len(forest.estimators_)))
kk=0
for t in forest.estimators_:
    print(kk)
    N15_std_global_mat[:,kk]=t.predict(X_global)
    kk=kk+1

# Estimate the area-weighted means of d15N for all ensembles
N15_mean_vect = np.nansum(N15_std_global_mat*land_frac_vect[:,np.newaxis],axis=0)/np.nansum(land_frac_vect)
#N15_mean_vect = np.nansum(N15_std_global_mat*total_input_vect[:,np.newaxis],axis=0)/np.nansum(total_input_vect)

# Estimate the mean, Std, and also the quantiles of d15N for the weighted values
soil_N15_mean = np.nanmean(N15_mean_vect)
soil_N15_std = np.nanvar(N15_mean_vect)**0.5
soil_N15_mean_q025 = np.nanquantile(N15_mean_vect,0.025)
soil_N15_mean_q50 = np.nanquantile(N15_mean_vect,0.50)
soil_N15_mean_q975 = np.nanquantile(N15_mean_vect,0.975)

print('delta Pland-Soil dN15:',soil_N15_mean,soil_N15_std,soil_N15_mean_q025,soil_N15_mean_q50,soil_N15_mean_q975)

# Estimate the Std, and quantiles of d15N for original ensembles
N15_std_global = np.var(N15_std_global_mat,axis=1)**0.5
N15_q025_global = np.nanquantile(N15_std_global_mat,0.025,axis=1,interpolation='lower')
N15_q50_global = np.nanquantile(N15_std_global_mat,0.50,axis=1,interpolation='lower')
N15_q975_global = np.nanquantile(N15_std_global_mat,0.975,axis=1,interpolation='lower')

# Mask out the NaN values
N15_std_global[mask_nanana[:,1]] = np.nan
N15_q025_global[mask_nanana[:,1]] = np.nan
N15_q50_global[mask_nanana[:,1]] = np.nan
N15_q975_global[mask_nanana[:,1]] = np.nan

# Mask out the Inf values
X_global[mask_inf] = np.nan

mask_large = (X_global==100000)
X_global[mask_large] = np.nan

# Reshape the vecters of std and quantiles into global maps
N15_std_mat = N15_std_global.reshape(3600,-1)
N15_std_mat_new = np.transpose(np.fliplr(N15_std_mat),(1,0))
N15_std_mat_new[mask_desert] = np.nan

N15_q025_mat = N15_q025_global.reshape(3600,-1)
N15_q025_mat_new = np.transpose(np.fliplr(N15_q025_mat),(1,0))
N15_q025_mat_new[mask_desert] = np.nan

N15_q50_mat = N15_q50_global.reshape(3600,-1)
N15_q50_mat_new = np.transpose(np.fliplr(N15_q50_mat),(1,0))
N15_q50_mat_new[mask_desert] = np.nan

N15_q975_mat = N15_q975_global.reshape(3600,-1)
N15_q975_mat_new = np.transpose(np.fliplr(N15_q975_mat),(1,0))
N15_q975_mat_new[mask_desert] = np.nan

########################################################################
############################## Data saving #############################
# Save the global maps, and also the quantiles
atts = dict(description = "Global maps of soil delta_N15",
            contact = "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)",
            date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            resolution = "Regular 0.01-degree")
RF_file = Dataset('*****************************************.nc','w')
RF_file.ncattrs(atts)
RF_file.createDimension('longitude',len(lon))
RF_file.createDimension('latitude',len(lat))
long = RF_file.createVariable('lon','d',('longitude',))
latit = RF_file.createVariable('lat','d',('latitude',))
long[:] = lon[:]
latit[:] = lat[:]

N15_lf = RF_file.createVariable('N15_foliar_soil','f',('longitude','latitude'))
N15_lf_std = RF_file.createVariable('N15_foliar_soil_std','f',('longitude','latitude'))
N15_lf_lowq = RF_file.createVariable('N15_foliar_soil_lowq','f',('longitude','latitude'))
N15_lf_uppq = RF_file.createVariable('N15_foliar_soil_uppq','f',('longitude','latitude'))
N15_lf_midd = RF_file.createVariable('N15_foliar_soil_midd','f',('longitude','latitude'))

N15_lf[:] = np.transpose(diff_foliar_soil[:],(1,0))
N15_lf_std[:] = np.transpose(N15_std_mat_new[:],(1,0))
N15_lf_lowq[:] = np.transpose(N15_q025_mat_new[:],(1,0))
N15_lf_uppq[:] = np.transpose(N15_q975_mat_new[:],(1,0))
N15_lf_midd[:] = np.transpose(N15_q50_mat_new[:],(1,0))

RF_file.close()
