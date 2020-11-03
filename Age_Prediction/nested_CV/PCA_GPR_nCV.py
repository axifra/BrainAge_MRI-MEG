import numpy as np
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.svm import SVR
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm
import itertools
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
import pdb
from load_data import load_data

# all_data = np.load('X_MRI_subcort.npy')
# # all_data = np.genfromtxt('Data/all_feats_updated.csv', delimiter=',', dtype=np.float64)
# # all_data = np.load('Data/subcortical_intensities.npy')
# Y_total = all_data[:,all_data.shape[1]-1]
# X_subcortical = stats.zscore(all_data[:,0:all_data.shape[1]-1])
# # X_WM = np.load('Data/WM_intensities.npz'); X_WM = stats.zscore(X_WM['x'])
# # X_CSF = np.load('Data/CSF_intensities.npz'); X_CSF = stats.zscore(X_CSF['x'])

# # X_area = stats.zscore(np.load('Data/X_area.npy'))
# # X_thickness = stats.zscore(np.load('Data/X_thickness.npy'))
# # X_volume = stats.zscore(np.load('Data/X_volume.npy'))
# # X_MRI = stats.zscore(np.load('Data/X_MRI.npy'))
# # Y_total = np.load('Data/y.npy').astype('float64')
# # X_total = np.concatenate((X_area,X_thickness,X_volume, X_MRI),axis=1)
# # X_total = X_MRI
# # X_total = np.concatenate((X_subcortical,X_WM,X_CSF),axis=1)
# X_total = X_subcortical
# print 'Data loading done (subcortical intensities GM) ' + str(X_total.shape)

def filename_to_write(gm,wm,cortical,subcortical,mri,psd,aec,ilc,meg,scorer):
	fname = 'Results_'+scorer+'_PCA_GPR'
	if gm and not mri:
		fname+='_GM'
	if wm and not mri:
		fname+='_WM'
	if cortical and not mri:
		fname+='_cort'
	if subcortical and not mri:
		fname+='_subcort'
	if mri:
		fname+='_mri'
	if psd or meg:
		fname+='_psd'
	if aec or meg:
		fname+='_aec'
	if ilc or meg:
		fname+='_ilc'
	fname+='.txt'
	return fname

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gm', help="binary (0/1) flag to include GM", default='0',type=int)
	parser.add_argument('--wm', help="binary (0/1) flag to include WM", default='0',type=int)
	parser.add_argument('--cort', help="binary (0/1) flag to include cortical GM", default='0',type=int)
	parser.add_argument('--subcort', help="binary (0/1) flag to include subcortical GM", default='0',type=int)
	parser.add_argument('--mri', help="binary (0/1) flag to include MRI", default='0',type=int)
	parser.add_argument('--psd', help="binary (0/1) flag to include PSD", default='0',type=int)
	parser.add_argument('--aec', help="binary (0/1) flag to include AEC", default='0',type=int)
	parser.add_argument('--ilc', help="binary (0/1) flag to include ILC", default='0',type=int)
	parser.add_argument('--meg', help="binary (0/1) flag to include MEG", default='0',type=int)
	parser.add_argument('--scorer', help="mae or r2, decides the scoring metric", default='mae')
	args = parser.parse_args()
	X_total, Y_total = load_data(gm=args.gm,wm=args.wm,cortical=args.cort,subcortical=args.subcort,mri=args.mri,psd=args.psd,aec=args.aec,ilc=args.ilc,meg=args.meg)

	scorer = 'neg_mean_absolute_error' if args.scorer=='mae' else 'r2'
	print(scorer)

	folds = 10
	np.random.seed(7)
	noise_level = np.logspace(-2,2,5)
	sigma_level = np.logspace(-2,2,5)
	length_level = np.logspace(-2,2,5)
	values_list = list(itertools.product(sigma_level,noise_level))
	kernel_list1 = [DotProduct(sigma_0=s)+WhiteKernel(noise_level=n) for s,n in values_list]
	values_list = list(itertools.product(length_level,noise_level))
	kernel_list2 = [RBF(length_scale=l)+WhiteKernel(noise_level=n) for l,n in values_list]
	# kernel_dict = dict(kernel=kernel_list1+kernel_list2)
	kernel_dict = dict(classification__kernel=kernel_list1)
	pca_components = 10 if (args.psd or args.aec or args.ilc or args.meg) else 5

	clf_p = Pipeline(steps=[
		('reduce_dim',PCA(n_components=pca_components)),
		('classification',GaussianProcessRegressor(random_state=0))
		])

	clf = GridSearchCV(estimator=clf_p, param_grid=kernel_dict,cv=5,scoring=scorer,n_jobs=10,verbose=1)
	# pdb.set_trace()
	# clf.fit(X_total,Y_total)
	# print(clf.cv_results_['mean_train_score'])
	# print(clf.cv_results_['mean_test_score'])
	# print(clf.best_score_)
	# print(clf.best_index_)
	# print(clf.best_estimator_)

	all_scores = []
	for f in tqdm(range(folds)):
		perm = np.random.permutation(X_total.shape[0])
		Y_total_f = Y_total[perm]
		X_total_f = X_total[perm]
		scores = cross_val_score(clf,X_total_f,Y_total_f,scoring=scorer,cv=10,n_jobs=1,verbose=2)
		all_scores.append(scores)
		print(scores)
		print(np.mean(scores))

	all_scores = np.array(all_scores)
	np.savetxt(filename_to_write(gm=args.gm,wm=args.wm,cortical=args.cort,subcortical=args.subcort,mri=args.mri,psd=args.psd,aec=args.aec,ilc=args.ilc,meg=args.meg,scorer=args.scorer),
		all_scores.ravel(),fmt='%5.8f')
