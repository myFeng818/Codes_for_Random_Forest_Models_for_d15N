from ncload import ncload
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

'''##################################################################################
####################################################################################
######################        Random Forest algorithm   ############################
######################        for d15N_soil             ############################
Finished by Maoyuan Feng (fengmy@pku.edu.cn)
Contact: "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)"
'''
# set the lontitude and latitude used in this code
lon = np.arange(-180,180,0.1)
lat = np.arange(-56,84,0.1)

# load the mask of desert grid cells
desert_file = ncload('Dersert_percent.nc')
desert_value = desert_file.get('new_data')

new_data = np.array(desert_value[:])
mask_desert = (new_data>=85) # mask out the grid cells when the percent of desert is >85%

# load the 16 predictors of RF model
global_data_path = '************************'

file_name_list = ['gpp1982_2015.nc','nhx1980_2016.nc','noy1980_2016.nc',\
                  'p_pet1980_2018.nc','pre1980_2018.nc','tmp1980_2018.nc',\
                  'BD.nc','CLAY.nc','SAND.nc','SILT.nc','OC.nc','TC.nc','TN.nc',\
                  'PHH2O.nc','am.nc','em.nc','nfix.nc']
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
# set the inf values to a large value 100000X_global[mask_inf] = 100000

X_global_pd = pd.DataFrame(X_global)
print(X_global_pd.isnull().any())
print(np.isnan(X_global).any())
print(np.isinf(X_global).any())
print(np.argwhere(np.isinf(X_global)))
#gpp_file = ncload(global_data_path+'gpp1982_2015_01d.nc')
#gpp_mat = gpp_file.get('tmp')
#BD_file = ncload(global_data_path+'BD.nc')
#BD_mat = BD_file.get('tmp')

# load the plant N15 observations
Craine_foliar_data = 'leafP_v1.xlsx'

foliar_data = pd.read_excel(Craine_foliar_data,header=0,index_col=0)
foliar_new=foliar_data.dropna()
# drop all nan/empty values

# calculate the TC/TN ratio values
foliar_rf_tc = foliar_new['TC'].values
foliar_rf_tn = foliar_new['TN'].values

foliar_new['C/N'] = np.where(foliar_rf_tn<1e-3,100000,foliar_rf_tc/foliar_rf_tn)

# keep all vects of the 16 predictors, and drop all the useless value
foliar_rf = foliar_new.drop(labels=['Latitude','Longitude','AWC_CLASS','TC','TN'],axis=1)
soil_rf = soil_new.drop(labels=['Latitude','Longitude','AWC_CLASS','TC','TN'],axis=1)

#feature_labels = foliar_rf.drop(labels=['N15','pet'],axis=1).columns
feature_labels = pd.Series(['BD','Clay','OC','pH','Sand','Silt','AM','EM','GPP','Nfix','NHx','NOy','P','T','P/ET','C/N'])


###################################################################################
############################# Training of the RF algorithm  #######################
X = (foliar_rf.drop(labels=['N15','pet'],axis=1)).values
Y = (foliar_rf['N15']).values

# Set up the Random Forest model
forest = RandomForestRegressor(n_estimators=1000,random_state=0,bootstrap=True,oob_score=True,min_samples_leaf=1,max_features='sqrt',n_jobs=4)
forest.fit(X,Y) # Train the model

# Predict the global map of d15N for plant
N15_global = np.zeros((5040000,))
N15_global = forest.predict(X_global)
N15_global[mask_nanana[:,1]] = np.nan

# Reshape the predicted vect into a global map
N15_mat = N15_global.reshape(3600,-1)
N15_mat_new = np.transpose(np.fliplr(N15_mat),(1,0))
N15_mat_new[mask_desert] = np.nan # mask out the NaN values

# Calculate all the ensembles predicted by the Trees of Random Forest model
N15_std_global_mat = np.zeros((5040000,len(forest.estimators_)))
kk=0
for t in forest.estimators_:
    print(kk)
    N15_std_global_mat[:,kk]=t.predict(X_global)
    kk=kk+1

# Calculate the SDs and quantiles
N15_std_global = np.var(N15_std_global_mat,axis=1)**0.5
N15_q025_global = np.nanquantile(N15_std_global_mat,0.025,axis=1,interpolation='lower')
N15_q50_global = np.nanquantile(N15_std_global_mat,0.50,axis=1,interpolation='lower')
N15_q975_global = np.nanquantile(N15_std_global_mat,0.975,axis=1,interpolation='lower')

# Mask out the NaN values
N15_std_global[mask_nanana[:,1]] = np.nan
N15_q025_global[mask_nanana[:,1]] = np.nan
N15_q50_global[mask_nanana[:,1]] = np.nan
N15_q975_global[mask_nanana[:,1]] = np.nan


X_global[mask_inf] = np.nan

mask_large = (X_global==100000)
X_global[mask_large] = np.nan

# Reshape the predicted vectors of SDs and quantiles into global maps
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
atts = dict(description = "Global maps of plant delta_N15",
            contact = "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)",
            date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            resolution = "Regular 0.01-degree")
RF_file = Dataset('RF_foliar_N15.nc','w')
RF_file.ncattrs(atts)
RF_file.createDimension('longitude',len(lon))
RF_file.createDimension('latitude',len(lat))
long = RF_file.createVariable('lon','d',('longitude',))
latit = RF_file.createVariable('lat','d',('latitude',))
long[:] = lon[:]
latit[:] = lat[:]

N15_lf = RF_file.createVariable('N15_foliar','f',('longitude','latitude'))
N15_lf_std = RF_file.createVariable('N15_foliar_std','f',('longitude','latitude'))
N15_lf_lowq = RF_file.createVariable('N15_foliar_lowq','f',('longitude','latitude'))
N15_lf_uppq = RF_file.createVariable('N15_foliar_uppq','f',('longitude','latitude'))
N15_lf_midd = RF_file.createVariable('N15_foliar_midd','f',('longitude','latitude'))

N15_lf[:] = np.transpose(N15_mat_new[:],(1,0))
N15_lf_std[:] = np.transpose(N15_std_mat_new[:],(1,0))
N15_lf_lowq[:] = np.transpose(N15_q025_mat_new[:],(1,0))
N15_lf_uppq[:] = np.transpose(N15_q975_mat_new[:],(1,0))
N15_lf_midd[:] = np.transpose(N15_q50_mat_new[:],(1,0))

RF_file.close()

