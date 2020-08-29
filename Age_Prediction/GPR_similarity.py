import numpy as np
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import cosine_similarity as similarity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR
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
X_psd = stats.zscore(np.load('../Data/X_psd_rel_LEAK.npy'))
# X_MRI_cort = stats.zscore(np.load('../Data/X_MRI_cort.npy'))
# X_area = stats.zscore(np.load('../../NEUR608/Data/X_area.npy'))
# X_thickness = stats.zscore(np.load('../../NEUR608/Data/X_thickness.npy'))
# X_volume = stats.zscore(np.load('../../NEUR608/Data/X_volume.npy'))
Y_total = np.load('../Data/y.npy').astype('float64')
X_total = np.concatenate((X_AEC_within,X_ILC,X_psd),axis=1)
X_total = X_psd
print 'Data loading done (PSD)' + str(X_total.shape)

np.random.seed(7)

avg_train_mae = 0;
avg_test_mae = 0;
avg_train_R2 = 0;
avg_test_R2 = 0;
folds = 20;
train_mae_arr = []
test_mae_arr = []
train_R2_arr = []
test_R2_arr = []

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

	X_training_similarity = np.dot(X_training,np.transpose(X_training))
	X_validation_similarity = np.dot(X_validation,np.transpose(X_training))
	X_all_similarity = np.append(X_training_similarity,X_validation_similarity,axis=0);
	X_all_similarity = stats.zscore(X_all_similarity);
	X_training_similarity = X_all_similarity[:500,:]
	X_validation_similarity = X_all_similarity[500:,:]
	#tqdm.write(str(X_training_similarity.shape)+','+ str(X_validation_similarity.shape))

	gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_training_similarity, Y_training)
	#gpr = SVR(kernel='rbf',C=1e3, gamma='auto').fit(X_training_similarity, Y_training)
	train_R2 = gpr.score(X_training_similarity,Y_training)
	test_R2 =  gpr.score(X_validation_similarity,Y_validation)
	train_mae = mae(Y_training,gpr.predict(X_training_similarity))
	test_mae = mae(Y_validation,gpr.predict(X_validation_similarity))
	#print("Training R^2 score:",gpr.score(X_training,Y_training))
	#print("Validation R^2 score:",gpr.score(X_validation,Y_validation))
	#print("Training MAE:",mae(Y_training,gpr.predict(X_training)))
	#print("Validation MAE:",mae(Y_validation,gpr.predict(X_validation)))
	train_mae_arr.append(train_mae)
	train_R2_arr.append(train_R2)
	test_mae_arr.append(test_mae)
	test_R2_arr.append(test_R2)


train_mae_arr = np.array(train_mae_arr)
train_R2_arr = np.array(train_R2_arr)
test_mae_arr = np.array(test_mae_arr)
test_R2_arr = np.array(test_R2_arr)

avg_train_mae = np.mean(train_mae_arr)
avg_test_mae = np.mean(test_mae_arr)
avg_train_R2 = np.mean(train_R2_arr)
avg_test_R2 = np.mean(test_R2_arr)

std_train_mae = np.std(train_mae_arr)/np.sqrt(folds)
std_test_mae = np.std(test_mae_arr)/np.sqrt(folds)
std_train_R2 = np.std(train_R2_arr)/np.sqrt(folds)
std_test_R2 = np.std(test_R2_arr)/np.sqrt(folds)

print "Average Training MAE: ", avg_train_mae, " +- ",std_train_mae
print "Average Training R2: ", avg_train_R2, " +- ",std_train_R2
print "Average Testing MAE: ", avg_test_mae, " +- ",std_test_mae
print "Average Testing R2: ", avg_test_R2, " +- ",std_test_R2
