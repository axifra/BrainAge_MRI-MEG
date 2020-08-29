import numpy as np
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm

kernel = DotProduct() + WhiteKernel()
concat = 0
meg_concat = 1

# all_data = np.genfromtxt('../Data/psd_All.csv', delimiter=',', dtype=np.float64)
# Y_total = all_data[:,all_data.shape[1]-1]
# X_total = stats.zscore(all_data[:,0:all_data.shape[1]-1])

freq_bands = 7
X_AEC = stats.zscore(np.load('../Data/X_AEC_within_LEAK.npy'))

# X_AEC_within = X_AEC[:,int(0*X_AEC.shape[1]/freq_bands):int(1*X_AEC.shape[1]/freq_bands)]	# adj matrix in delta band
# X_AEC_within = X_AEC[:,int(1*X_AEC.shape[1]/freq_bands):int(2*X_AEC.shape[1]/freq_bands)]	# adj matrix in theta band
# X_AEC_within = (X_AEC[:,int(2*X_AEC.shape[1]/freq_bands):int(3*X_AEC.shape[1]/freq_bands)]+X_AEC[:,int(3*X_AEC.shape[1]/freq_bands):int(4*X_AEC.shape[1]/freq_bands)])/2	# averaging adj matrix over mu and alpha bands
# X_AEC_within = (X_AEC[:,int(4*X_AEC.shape[1]/freq_bands):int(5*X_AEC.shape[1]/freq_bands)]+X_AEC[:,int(5*X_AEC.shape[1]/freq_bands):int(6*X_AEC.shape[1]/freq_bands)])/2	# averaging adj matrix over lower and higher beta bands
# X_AEC_within = X_AEC[:,int(6*X_AEC.shape[1]/freq_bands):int(7*X_AEC.shape[1]/freq_bands)]	# adj matrix in gamma band

X_ILC = stats.zscore(np.load('../Data/X_ILC_LEAK.npy'))
# X_MRI = stats.zscore(np.load('../Data/X_MRI.npy'))
# X_MRI_intensity_feat = stats.zscore(np.load('../Data/X_MRI_parcel_int_reordered.npy')).reshape(len(X_AEC_within),-1)
X_MRI_cort = stats.zscore(np.load('../Data/X_MRI_cort.npy'))
X_psd = stats.zscore(np.load('../Data/X_psd_rel_LEAK.npy'))
# X_area = stats.zscore(np.load('../../NEUR608/Data/X_area.npy'))
# X_thickness = stats.zscore(np.load('../../NEUR608/Data/X_thickness.npy'))
# X_volume = stats.zscore(np.load('../../NEUR608/Data/X_volume.npy'))
Y_total = np.load('../Data/y.npy').astype('float64')
X_total_meg = np.concatenate((X_AEC,X_ILC,X_psd),axis=1)
# X_total_meg = X_psd
X_total_mri = X_MRI_cort
# X_total = np.concatenate((X_AEC_within,X_ILC,X_psd,X_MRI),axis=1)
# X_total_mri = np.concatenate((X_area,X_thickness,X_volume),axis=1)
# X_total = np.concatenate((X_AEC_within,X_ILC,X_psd,X_area,X_thickness,X_volume),axis=1)
# X_total = np.concatenate((X_MRI_intensity_feat,X_psd,X_area,X_thickness,X_volume),axis=1)
X_total = np.concatenate((X_total_mri,X_total_meg),axis=1)
# X_total = X_AEC_within
print 'Data loading done (MRI_cort+PSD+AEC+ILC)' + str(X_total.shape) + ' with concat='+str(concat) + (', meg_concat='+str(meg_concat) if not concat else '')

np.random.seed(7)

avg_train_mae = 0;
avg_test_mae = 0;
avg_train_R2 = 0;
avg_test_R2 = 0;
folds = 20;

for f in tqdm(range(folds)):
	if concat:
		perm = np.random.permutation(X_total.shape[0])
		Y_total_f = Y_total[perm]
		X_total_f = X_total[perm]

		X_training = X_total_f[:500,:]
		X_validation = X_total_f[500:,:]

		Y_training = Y_total_f[:500]
		Y_validation = Y_total_f[500:]

		cca = CCA(n_components=1).fit(X_training,Y_training)
		X_training = cca.transform(X_training)
		X_validation = cca.transform(X_validation)

	else:
		if meg_concat:
			perm = np.random.permutation(X_total.shape[0])
			Y_total_f = Y_total[perm]
			X_total_meg_f = X_total_meg[perm]
			X_total_mri_f = X_total_mri[perm]

			X_training_meg = X_total_meg_f[:500,:]
			X_training_mri = X_total_mri_f[:500,:]
			X_validation_meg = X_total_meg_f[500:,:]
			X_validation_mri = X_total_mri_f[500:,:]

			Y_training = Y_total_f[:500]
			Y_validation = Y_total_f[500:]

			cca = CCA(n_components=1).fit(X_training_meg,Y_training)
			X_training_meg = cca.transform(X_training_meg)
			X_validation_meg = cca.transform(X_validation_meg)
			cca = CCA(n_components=1).fit(X_training_mri,Y_training)
			X_training_mri = cca.transform(X_training_mri)
			X_validation_mri = cca.transform(X_validation_mri)
			X_training = np.concatenate((X_training_meg,X_training_mri),axis=1)
			X_validation = np.concatenate((X_validation_meg,X_validation_mri),axis=1)

		else:
			perm = np.random.permutation(X_total.shape[0])
			Y_total_f = Y_total[perm]
			X_total_AEC_f = X_AEC_within[perm]
			X_total_ILC_f = X_ILC[perm]
			X_total_psd_f = X_psd[perm]
			X_total_mri_f = X_total_mri[perm]

			X_training_AEC = X_total_AEC_f[:500,:]
			X_training_ILC = X_total_ILC_f[:500,:]
			X_training_psd = X_total_psd_f[:500,:]
			X_training_mri = X_total_mri_f[:500,:]
			X_validation_AEC = X_total_AEC_f[500:,:]
			X_validation_ILC = X_total_ILC_f[500:,:]
			X_validation_psd = X_total_psd_f[500:,:]
			X_validation_mri = X_total_mri_f[500:,:]

			Y_training = Y_total_f[:500]
			Y_validation = Y_total_f[500:]

			cca = CCA(n_components=1).fit(X_training_AEC,Y_training)
			X_training_AEC = cca.transform(X_training_AEC)
			X_validation_AEC = cca.transform(X_validation_AEC)
			cca = CCA(n_components=1).fit(X_training_ILC,Y_training)
			X_training_ILC = cca.transform(X_training_ILC)
			X_validation_ILC = cca.transform(X_validation_ILC)
			cca = CCA(n_components=1).fit(X_training_psd,Y_training)
			X_training_psd = cca.transform(X_training_psd)
			X_validation_psd = cca.transform(X_validation_psd)
			cca = CCA(n_components=1).fit(X_training_mri,Y_training)
			X_training_mri = cca.transform(X_training_mri)
			X_validation_mri = cca.transform(X_validation_mri)
			X_training = np.concatenate((X_training_AEC,X_training_ILC,X_training_psd,X_training_mri),axis=1)
			X_validation = np.concatenate((X_validation_AEC,X_validation_ILC,X_validation_psd,X_validation_mri),axis=1)

	age_mean  = np.mean(Y_training)
	Y_training = Y_training - age_mean
	Y_validation = Y_validation - age_mean

	gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_training, Y_training)
	#gpr = SVR(kernel='rbf',C=1e3, gamma='auto').fit(X_training, Y_training)
	avg_train_R2 = avg_train_R2+ gpr.score(X_training,Y_training)
	avg_test_R2 = avg_test_R2+ gpr.score(X_validation,Y_validation)
	avg_train_mae = avg_train_mae + mae(Y_training,gpr.predict(X_training))
	avg_test_mae = avg_test_mae + mae(Y_validation,gpr.predict(X_validation))

avg_train_mae = avg_train_mae/folds
avg_test_mae = avg_test_mae/folds
avg_train_R2 = avg_train_R2/folds
avg_test_R2 = avg_test_R2/folds

print "Average Training MAE: ", avg_train_mae
print "Average Training R2: ", avg_train_R2
print "Average Testing MAE: ", avg_test_mae
print "Average Testing R2: ", avg_test_R2
