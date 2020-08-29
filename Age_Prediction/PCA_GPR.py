import numpy as np
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm

kernel = DotProduct() + WhiteKernel()

# all_data = np.genfromtxt('../Data/psd_All.csv', delimiter=',', dtype=np.float64)
# Y_total = all_data[:,all_data.shape[1]-1]
# X_total = stats.zscore(all_data[:,0:all_data.shape[1]-1])

X_AEC_within = stats.zscore(np.load('../Data/X_AEC_within_LEAK.npy'))
X_ILC = stats.zscore(np.load('../Data/X_ILC_LEAK.npy'))
# X_MRI = stats.zscore(np.load('../Data/X_MRI.npy'))
# X_MRI_cort = stats.zscore(np.load('../Data/X_MRI_cort.npy'))
X_psd = stats.zscore(np.load('../Data/X_psd_rel_LEAK.npy'))
# X_area = stats.zscore(np.load('../../NEUR608/Data/X_area.npy'))
# X_thickness = stats.zscore(np.load('../../NEUR608/Data/X_thickness.npy'))
# X_volume = stats.zscore(np.load('../../NEUR608/Data/X_volume.npy'))
Y_total = np.load('../Data/y.npy').astype('float64')
X_total = np.concatenate((X_AEC_within,X_ILC,X_psd),axis=1)
# X_total = np.concatenate((X_AEC_within,X_ILC,X_psd,X_area,X_thickness,X_volume),axis=1)
# X_total = X_AEC_within
print 'Data loading done (AEC+ILC+PSD)' + str(X_total.shape)

pca = PCA().fit(X_total)
# np.savetxt(os.path.join('Data','GM_PCA_varExp.csv'),pca.explained_variance_ratio_,delimiter=',')
plt.plot(pca.explained_variance_ratio_[:100])
plt.show()
plt.plot(np.cumsum(pca.explained_variance_ratio_[:100]))
plt.show()

np.random.seed(7)

avg_train_mae = 0;
avg_test_mae = 0;
avg_train_R2 = 0;
avg_test_R2 = 0;
folds = 20;

for f in tqdm(range(folds)):
	perm = np.random.permutation(X_total.shape[0])
	Y_total_f = Y_total[perm]
	X_total_f = X_total[perm]
	# X_total_mri_f = X_MRI[perm]

	# X_training_meg = X_total_meg_f[:500,:]
	# X_training_mri = X_total_mri_f[:500,:]
	# X_validation_meg = X_total_meg_f[500:,:]
	# X_validation_mri = X_total_mri_f[500:,:]
	X_training = X_total_f[:500,:]
	X_validation = X_total_f[500:,:]

	Y_training = Y_total_f[:500]
	Y_validation = Y_total_f[500:]

	# pca = PCA(n_components=10).fit(X_training_meg)
	# X_training_meg = pca.transform(X_training_meg)
	# X_validation_meg = pca.transform(X_validation_meg)
	# pca = PCA(n_components=5).fit(X_training_mri)
	# X_training_mri = pca.transform(X_training_mri)
	# X_validation_mri = pca.transform(X_validation_mri)
	# X_training = np.concatenate((X_training_meg,X_training_mri),axis=1)
	# X_validation = np.concatenate((X_validation_meg,X_validation_mri),axis=1)
	pca = PCA(n_components=10).fit(X_training)
	X_training = pca.transform(X_training)
	X_validation = pca.transform(X_validation)

	age_mean  = np.mean(Y_training)
	Y_training = Y_training - age_mean
	Y_validation  =Y_validation - age_mean

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
