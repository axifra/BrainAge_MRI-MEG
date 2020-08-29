import numpy as np
import sys
from sklearn.cross_decomposition import CCA
from sklearn.utils import resample
import matplotlib.pyplot as plt
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm

# all_data = np.genfromtxt('../Data/psd_All.csv', delimiter=',', dtype=np.float64)
# Y_total = all_data[:,all_data.shape[1]-1]
# X_total = stats.zscore(all_data[:,0:all_data.shape[1]-1])

X_AEC_within = stats.zscore(np.load('../Data/X_AEC_within_LEAK.npy'))
X_ILC = stats.zscore(np.load('../Data/X_ILC_LEAK.npy'))
X_psd = stats.zscore(np.load('../Data/X_psd_rel_LEAK.npy'))

Y_total = np.load('../Data/y.npy').astype('float64')
X_total = np.concatenate((X_AEC_within,X_ILC,X_psd),axis=1)

np.random.seed(7)

# check_loadings_arr = []
permutations = 1000;
folds = 100;
repeatitions = permutations/folds

for r in range(repeatitions):
	loadings_arr = []
	for f in tqdm(range(folds)):
		try:
			X_total_f, Y_total_f = resample(X_total,Y_total)

			cca = CCA(n_components=1).fit(X_total_f,Y_total_f)
		except np.linalg.linalg.LinAlgError:
			print 'Failed to converge SVD in fold ' + str(f)
		except:
			print 'Some other error in fold ' + str(f)
		else:
			loadings_arr.append((cca.y_loadings_*cca.x_loadings_).reshape(cca.x_loadings_.shape[0],))
			# check_loadings_arr.append((cca.y_loadings_*cca.x_loadings_).reshape(cca.x_loadings_.shape[0],))

	loadings_arr = np.array(loadings_arr)
	loadings_arr_mean = np.mean(loadings_arr,axis=0)
	loadings_arr_sd = np.std(loadings_arr,axis=0) #/np.sqrt(loadings_arr.shape[0])	# calculates standard error
	if r==0:
		running_std = loadings_arr_sd
		running_mean = loadings_arr_mean
		running_count = folds

	else:
		running_std = np.sqrt((running_count*np.square(running_std) + folds*np.square(loadings_arr_sd) + running_count*folds*(np.square(running_mean) + np.square(loadings_arr_mean) - 2*running_mean*loadings_arr_mean)/(running_count+folds))/(running_count+folds))
		running_mean = (running_count*running_mean + folds*loadings_arr_mean)/(running_count+folds)
		running_count += folds

loadings_arr_se = running_std/np.sqrt(permutations)
loadings_arr_se = loadings_arr_se.reshape(loadings_arr_se.shape[0],1)
# check_loadings_arr = np.array(check_loadings_arr)
print loadings_arr.shape
print loadings_arr_se.shape

cca_true = CCA(n_components=1).fit(X_total,Y_total)
print(cca_true.y_loadings_)
true_loadings = cca_true.y_loadings_*cca_true.x_loadings_
print true_loadings.shape

corrected_loadings = np.divide(true_loadings,loadings_arr_se)
print corrected_loadings.shape

all_loadings =  np.array([true_loadings,loadings_arr_se,corrected_loadings])
all_loadings = all_loadings.reshape(all_loadings.shape[0],all_loadings.shape[1])
print(all_loadings.shape)

np.save(os.path.join('../Data','MEG_feats_BS_loadings_beam_LEAK_1000.npy'),all_loadings)
