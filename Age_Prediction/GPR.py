import numpy as np
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm

kernel = DotProduct() + WhiteKernel()

def thresh_adjacency_matrix(adj, method='Percentile', percentile=90):
	if method=='Percentile':
		thresh_vals = [np.percentile(adj[subj],percentile) for subj in range(adj.shape[0])]

	return thresh_vals

# all_data = np.genfromtxt('../Data/psd_All.csv', delimiter=',', dtype=np.float64)
# Y_total = all_data[:,all_data.shape[1]-1]
# X_total = stats.zscore(all_data[:,0:all_data.shape[1]-1])

freq_bands = 7
X_AEC = stats.zscore(np.load('../Data/X_AEC_within_LEAK.npy'))
# X_AEC_within = X_AEC
X_AEC_within = X_AEC[:,int(0*X_AEC.shape[1]/freq_bands):int(1*X_AEC.shape[1]/freq_bands)]	# adj matrix in delta band
# X_AEC_within = X_AEC[:,int(1*X_AEC.shape[1]/freq_bands):int(2*X_AEC.shape[1]/freq_bands)]	# adj matrix in theta band
# X_AEC_within = (X_AEC[:,int(2*X_AEC.shape[1]/freq_bands):int(3*X_AEC.shape[1]/freq_bands)]+X_AEC[:,int(3*X_AEC.shape[1]/freq_bands):int(4*X_AEC.shape[1]/freq_bands)])/2	# averaging adj matrix over mu and alpha bands
# X_AEC_within = (X_AEC[:,int(4*X_AEC.shape[1]/freq_bands):int(5*X_AEC.shape[1]/freq_bands)]+X_AEC[:,int(5*X_AEC.shape[1]/freq_bands):int(6*X_AEC.shape[1]/freq_bands)])/2	# averaging adj matrix over lower and higher beta bands
# X_AEC_within = X_AEC[:,int(6*X_AEC.shape[1]/freq_bands):int(7*X_AEC.shape[1]/freq_bands)]	# adj matrix in gamma band

Num_nodes = int((1+np.sqrt(X_AEC_within.shape[1]*8+1))/2)
thresh_vals = thresh_adjacency_matrix(X_AEC_within,'Percentile',95)
# temp = np.array([X_AEC_within[subj]*(X_AEC_within[subj]>thresh_vals[subj]) for subj in range(X_AEC_within.shape[0])])
temp = np.array([X_AEC_within[subj][X_AEC_within[subj]>thresh_vals[subj]] for subj in range(X_AEC_within.shape[0])])
self_loops = np.repeat(np.max(temp,1)[:,np.newaxis],Num_nodes,1)
X_AEC_thresh = temp #np.concatenate((temp,self_loops),axis=1)

X_ILC = stats.zscore(np.load('../Data/X_ILC_LEAK.npy'))
# X_MRI = stats.zscore(np.load('../Data/X_MRI.npy'))
X_MRI_intensity_feat = stats.zscore(np.load('../Data/X_MRI_parcel_int_reordered.npy')).reshape(len(X_AEC_within),-1)
# X_MRI_cort = stats.zscore(np.load('../Data/X_MRI_cort.npy'))
X_psd = stats.zscore(np.load('../Data/X_psd_rel_LEAK.npy'))
X_area = stats.zscore(np.load('../Data/X_area_reordered.npy'))
X_thickness = stats.zscore(np.load('../Data/X_thickness_reordered.npy'))
X_volume = stats.zscore(np.load('../Data/X_volume_reordered.npy'))
Y_total = np.load('../Data/y.npy').astype('float64')
# X_total = np.concatenate((X_AEC_within,X_ILC,X_psd,X_MRI),axis=1)
# X_total = np.concatenate((X_AEC_within,X_ILC,X_psd,X_area,X_thickness,X_volume),axis=1)
# X_total = np.concatenate((X_MRI_intensity_feat,X_AEC_within,X_psd,X_area,X_thickness,X_volume),axis=1)
# X_total = np.concatenate((X_MRI_intensity_feat,X_AEC_thresh,X_psd,X_area,X_thickness,X_volume),axis=1)
X_total = np.concatenate((X_MRI_intensity_feat,X_AEC_thresh,X_psd,X_area,X_thickness,X_volume),axis=1)
# X_total = X_AEC_thresh
print 'Data loading done (MRI_mean_std+AEC_thresh(delta)+PSD+cATV)' + str(X_total.shape)	#

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

	X_training = X_total_f[:500,:]
	X_validation = X_total_f[500:,:]

	Y_training = Y_total_f[:500]
	Y_validation = Y_total_f[500:]

	age_mean  = np.mean(Y_training)
	Y_training = Y_training - age_mean
	Y_validation  =Y_validation - age_mean

	gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_training, Y_training)
	#gpr = SVR(kernel='rbf',C=1e2, gamma='auto').fit(X_training, Y_training)
	avg_train_R2 = avg_train_R2+ gpr.score(X_training,Y_training)
	avg_test_R2 = avg_test_R2+ gpr.score(X_validation,Y_validation)
	avg_train_mae = avg_train_mae + mae(Y_training,gpr.predict(X_training))
	avg_test_mae = avg_test_mae + mae(Y_validation,gpr.predict(X_validation))
	#print("Training R^2 score:",gpr.score(X_training,Y_training))
	#print("Validation R^2 score:",gpr.score(X_validation,Y_validation))
	#print("Training MAE:",mae(Y_training,gpr.predict(X_training)))
	#print("Validation MAE:",mae(Y_validation,gpr.predict(X_validation)))

avg_train_mae = avg_train_mae/folds
avg_test_mae = avg_test_mae/folds
avg_train_R2 = avg_train_R2/folds
avg_test_R2 = avg_test_R2/folds

print "Average Training MAE: ", avg_train_mae
print "Average Training R2: ", avg_train_R2
print "Average Testing MAE: ", avg_test_mae
print "Average Testing R2: ", avg_test_R2
