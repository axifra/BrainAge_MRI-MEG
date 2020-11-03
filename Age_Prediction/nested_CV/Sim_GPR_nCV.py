# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:32:43 2020

@author: Alba
"""

import numpy as np
import sys
from multiprocessing import Pool
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import cosine_similarity as similarity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.svm import SVR
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
import pdb
from load_data import load_data


##

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
np.random.seed(7)
noise_level = np.logspace(-2,2,5)
sigma_level = np.logspace(-2,2,5)
length_level = np.logspace(-2,2,5)
values_list = list(itertools.product(sigma_level,noise_level))
kernel_list1 = [DotProduct(sigma_0=s)+WhiteKernel(noise_level=n) for s,n in values_list]
kernel_dict = dict(classification__kernel=kernel_list1)
##
    
def filename_to_write(gm,wm,cortical,subcortical,mri,psd,aec,ilc,meg,scorer):
	fname = 'Results_'+scorer+'_Sim_GPR'
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

def Sim_GPR_nCV_fold(fold):
    global X_total
    global Y_total
    global kernel_dict
    outer_folds = 10
    inner_folds = 5
    scores_mae_f = []
    scores_r2_f = []
    np.random.seed(fold)
    perm = np.random.permutation(X_total.shape[0])
    X_total_f = X_total[perm]
    Y_total_f = Y_total[perm]
    ss_outer = ShuffleSplit(n_splits=outer_folds, test_size=1/outer_folds, random_state=0)
    for train_val_index, test_index in ss_outer.split(X_total_f,Y_total_f):
        X_train_val = X_total_f[train_val_index]
        Y_train_val = Y_total_f[train_val_index]
        X_test = X_total_f[test_index]
        Y_test = Y_total_f[test_index]
        best_mae = 1000
        best_mae_kernel = None
        best_r2 = 0
        best_r2_kernel = None
        for kernel in kernel_dict['classification__kernel']:
            score_mae = 0
            score_r2 = 0
            ss_inner = ShuffleSplit(n_splits=inner_folds, test_size=1/inner_folds, random_state=0)
            for train_index, val_index in ss_inner.split(X_train_val,Y_train_val):
                X_train = X_train_val[train_index]
                Y_train = Y_train_val[train_index]
                X_val = X_train_val[val_index]
                Y_val = Y_train_val[val_index]
                age_mean = np.mean(Y_train)
                Y_train = Y_train-age_mean
                Y_val = Y_val-age_mean
                X_train_sim = np.dot(X_train,np.transpose(X_train))
                X_val_sim = np.dot(X_val,np.transpose(X_train))
                gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train_sim,Y_train)
                Y_predict = gpr.predict(X_val_sim)
                if args.scorer == 'mae' or args.scorer == 'both':
                    score_mae += mae(Y_val,Y_predict)
                if args.scorer == 'r2' or args.scorer == 'both':
                    score_r2 += r2_score(Y_val,Y_predict)
            score_mae /= inner_folds
            score_r2 /= inner_folds
            if (args.scorer == 'mae' or args.scorer == 'both') and score_mae<best_mae:
                best_mae = score_mae
                best_mae_kernel = kernel
            if (args.scorer == 'r2' or args.scorer == 'both') and score_r2>best_r2:
                best_r2 = score_r2
                best_r2_kernel = kernel
        
        age_mean = np.mean(Y_train_val)
        Y_train_val = Y_train_val-age_mean
        Y_test = Y_test-age_mean
        X_train_val_sim = np.dot(X_train_val,np.transpose(X_train_val))
        X_test_sim = np.dot(X_test,np.transpose(X_train_val))
        if (args.scorer == 'mae' or args.scorer == 'both'):
            gpr_mae = GaussianProcessRegressor(kernel=best_mae_kernel,random_state=0).fit(X_train_val_sim,Y_train_val)
            Y_predict = gpr_mae.predict(X_test_sim)
            scores_mae_f.append(mae(Y_test,Y_predict))
        if (args.scorer == 'r2' or args.scorer == 'both'):
            gpr_r2 = GaussianProcessRegressor(kernel=best_r2_kernel,random_state=0).fit(X_train_val_sim,Y_train_val)
            Y_predict = gpr_r2.predict(X_test_sim)
            scores_r2_f.append(r2_score(Y_test,Y_predict))
            
    scores = np.array([scores_mae_f,scores_r2_f])
        
    return scores

if __name__=='__main__':
    
    folds = 10
    all_scores_mae = []
    all_scores_r2 = []
    
    with Pool(processes=10) as pool:
        res = list(tqdm(pool.imap(Sim_GPR_nCV_fold, list(range(folds))),total=folds))
        
    res = np.array(res)        
    all_scores_mae = np.ravel(res[:,0,:])
    all_scores_r2 = np.ravel(res[:,1,:])

    if (args.scorer == 'mae' or args.scorer == 'both'):
        np.savetxt(filename_to_write(gm=args.gm,wm=args.wm,cortical=args.cort,subcortical=args.subcort,mri=args.mri,psd=args.psd,aec=args.aec,ilc=args.ilc,meg=args.meg,scorer='mae'),
    		all_scores_mae,fmt='%5.8f')
    if (args.scorer == 'r2' or args.scorer == 'both'):
        np.savetxt(filename_to_write(gm=args.gm,wm=args.wm,cortical=args.cort,subcortical=args.subcort,mri=args.mri,psd=args.psd,aec=args.aec,ilc=args.ilc,meg=args.meg,scorer='r2'),
    		all_scores_r2,fmt='%5.8f')