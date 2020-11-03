# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:54:59 2020

@author: Alba
"""

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import argparse
import itertools

##
parser = argparse.ArgumentParser()
parser.add_argument('--scorer', help="mae or r2, decides the scoring metric", default='both')
args = parser.parse_args()

data = np.load('X_MRI.npy')
X_MRI = stats.zscore(data[:,0:data.shape[1]-1])
Y_total = data[:,data.shape[1]-1]
data = np.load('X_psd_rel_LEAK.npy')
X_psd = stats.zscore(data)
data = np.load('X_AEC_within_LEAK.npy')
X_aec = stats.zscore(data)
data = np.load('X_ILC_LEAK.npy')
X_ilc = stats.zscore(data)
X_meg = np.concatenate((X_psd,X_aec,X_ilc),axis=1)

np.random.seed(7)
noise_level = np.logspace(-2,2,5)
sigma_level = np.logspace(-2,2,5)
length_level = np.logspace(-2,2,5)
values_list = list(itertools.product(sigma_level,noise_level))
kernel_list1 = [DotProduct(sigma_0=s)+WhiteKernel(noise_level=n) for s,n in values_list]
kernel_dict = dict(classification__kernel=kernel_list1)
n_estimators = [10,50,100]
depths = [5,10,20,None]
rf_params_list = list(itertools.product(n_estimators,depths))
##

def filename_to_write(scorer):
	fname = 'Results_'+scorer+'_CCA2_GPR_Stacking.txt'
	return fname

def CCA_GPR_Stacking_nCV_fold(fold):
    global X_MRI
    global X_meg
    global Y_total
    global kernel_dict
    outer_folds = 10
    inner_folds = 5
    scores_mae_f = []
    scores_r2_f = []
    Y_pred_array = []
    Y_chron_age_array = []
    np.random.seed(fold)
    perm = np.random.permutation(X_MRI.shape[0])
    X_MRI_f = X_MRI[perm]
    X_meg_f = X_meg[perm]
    Y_total_f = Y_total[perm]
    #ss_outer = ShuffleSplit(n_splits=outer_folds, test_size=1/outer_folds, random_state=0)
    ss_outer = KFold(n_splits=outer_folds, random_state=0)
    for train_val_index, test_index in tqdm(ss_outer.split(X_MRI_f)):
        X_MRI_train_val = X_MRI_f[train_val_index]
        X_meg_train_val = X_meg_f[train_val_index]
        Y_train_val = Y_total_f[train_val_index]
        X_MRI_test = X_MRI_f[test_index]
        X_meg_test = X_meg_f[test_index]
        Y_test = Y_total_f[test_index]
        best_MRI_mae = 1000
        best_kernel_MRI_mae = None
        best_meg_mae = 1000
        best_kernel_meg_mae = None
        best_MRI_r2 = -1000
        best_kernel_MRI_r2 = None
        best_meg_r2 = -1000
        best_kernel_meg_r2 = None        
        best_Y_predict_MRI_mae = None
        best_Y_predict_meg_mae = None
        best_Y_predict_MRI_r2 = None
        best_Y_predict_meg_r2 = None         
        # Hyper-parameter selection for GPR
        for kernel in kernel_dict['classification__kernel']:
            score_MRI_mae = 0
            score_MRI_r2 = 0
            score_meg_mae = 0
            score_meg_r2 = 0         
            Y_predict_MRI_f = []
            std_MRI_f = []
            Y_predict_meg_f = []
            std_meg_f = []
            #ss_inner = ShuffleSplit(n_splits=inner_folds, test_size=1/inner_folds, random_state=0)
            ss_inner = KFold(n_splits=inner_folds, random_state=0)
            # Inner-loop
            for train_index, val_index in ss_inner.split(X_MRI_train_val):
                X_MRI_train = X_MRI_train_val[train_index]
                X_meg_train = X_meg_train_val[train_index]
                Y_train = Y_train_val[train_index]
                X_MRI_val = X_MRI_train_val[val_index]
                X_meg_val = X_meg_train_val[val_index]
                Y_val = Y_train_val[val_index]
                age_mean = np.mean(Y_train)
                Y_train = Y_train-age_mean
                Y_val = Y_val-age_mean
                cca_MRI = CCA(n_components=1).fit(X_MRI_train,Y_train)
                cca_meg = CCA(n_components=1).fit(X_meg_train,Y_train)
                X_MRI_train = cca_MRI.transform(X_MRI_train)
                X_meg_train = cca_meg.transform(X_meg_train)
                X_MRI_val = cca_MRI.transform(X_MRI_val)
                X_meg_val = cca_meg.transform(X_meg_val)
                gpr_MRI = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_MRI_train,Y_train)
                gpr_meg = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_meg_train,Y_train)
                Y_predict_MRI, std_MRI = gpr_MRI.predict(X_MRI_val, return_std=True)
                Y_predict_MRI_f.extend(Y_predict_MRI)
                std_MRI_f.extend(std_MRI)
                Y_predict_meg, std_meg = gpr_meg.predict(X_meg_val, return_std=True)
                Y_predict_meg_f.extend(Y_predict_meg)
                std_meg_f.extend(std_meg)
                if args.scorer == 'mae' or args.scorer == 'both':
                    score_MRI_mae += mae(Y_val,Y_predict_MRI)
                    score_meg_mae += mae(Y_val,Y_predict_meg)
                if args.scorer == 'r2' or args.scorer == 'both':
                    score_MRI_r2 += r2_score(Y_val,Y_predict_MRI)
                    score_meg_r2 += r2_score(Y_val,Y_predict_meg)   
                    
            score_MRI_mae /= inner_folds
            score_meg_mae /= inner_folds
            score_MRI_r2 /= inner_folds
            score_meg_r2 /= inner_folds
            if (args.scorer == 'mae' or args.scorer == 'both'):
                if score_MRI_mae<best_MRI_mae:
                    best_MRI_mae = score_MRI_mae
                    best_kernel_MRI_mae = kernel
                    best_Y_predict_MRI_mae = np.stack((Y_predict_MRI_f,std_MRI_f),axis=1)
                if score_meg_mae<best_meg_mae:  
                    best_meg_mae = score_meg_mae
                    best_kernel_meg_mae = kernel
                    best_Y_predict_meg_mae = np.stack((Y_predict_meg_f,std_meg_f),axis=1)
            if (args.scorer == 'r2' or args.scorer == 'both'):
                if score_MRI_r2>best_MRI_r2:
                    best_MRI_r2 = score_MRI_r2
                    best_kernel_MRI_r2 = kernel
                    best_Y_predict_MRI_r2 = np.stack((Y_predict_MRI_f,std_MRI_f),axis=1)
                if score_meg_r2>best_meg_r2:  
                    best_meg_r2 = score_meg_r2
                    best_kernel_meg_r2 = kernel
                    best_Y_predict_meg_r2 = np.stack((Y_predict_meg_f,std_meg_f),axis=1)
        
        with open("test.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, np.array([best_MRI_mae,best_meg_mae,best_MRI_r2,best_meg_r2]))        
        
        # Hyper-parameter selection for stacking model (Random forests)
        best_mae = 1000
        best_r2 = -1000
        best_rf_params_mae = None
        best_rf_params_r2 = None
        for n_est,depth in rf_params_list:
            # MAE
            if (args.scorer == 'mae' or args.scorer == 'both'):
                best_Y_predict = np.concatenate((best_Y_predict_MRI_mae,best_Y_predict_meg_mae),axis=1) 
                score_mae = 0
                for train_index, val_index in ss_inner.split(X_MRI_train_val):
                    rf_mae = RandomForestRegressor(n_estimators=n_est,max_depth=depth).fit(best_Y_predict[train_index],Y_train_val[train_index])
                    Y_predict_stacked = rf_mae.predict(best_Y_predict[val_index])
                    score_mae += mae(Y_train_val[val_index],Y_predict_stacked)
                score_mae /=inner_folds
                if score_mae<best_mae:
                    best_mae = score_mae
                    best_rf_params_mae = (n_est,depth)
            # r2        
            if (args.scorer == 'r2' or args.scorer == 'both'):
                best_Y_predict = np.concatenate((best_Y_predict_MRI_r2,best_Y_predict_meg_r2),axis=1) 
                score_r2 = 0
                for train_index, val_index in ss_inner.split(X_MRI_train_val):
                    rf_r2 = RandomForestRegressor(n_estimators=n_est,max_depth=depth).fit(best_Y_predict[train_index],Y_train_val[train_index])
                    Y_predict_stacked = rf_r2.predict(best_Y_predict[val_index])
                    score_r2 += r2_score(Y_train_val[val_index],Y_predict_stacked)
                score_r2 /=inner_folds
                if score_r2>best_r2:
                    best_r2 = score_r2
                    best_rf_params_r2 = (n_est,depth)
     
        # Test set
        age_mean = np.mean(Y_train_val)
        Y_train_val = Y_train_val-age_mean
        Y_test = Y_test-age_mean
        cca_MRI = CCA(n_components=1).fit(X_MRI_train_val,Y_train_val)
        cca_meg = CCA(n_components=1).fit(X_meg_train_val,Y_train_val)
        X_MRI_train_val = cca_MRI.transform(X_MRI_train_val)
        X_meg_train_val = cca_meg.transform(X_meg_train_val)        
        X_MRI_test = cca_MRI.transform(X_MRI_test)
        X_meg_test = cca_meg.transform(X_meg_test)        
        if (args.scorer == 'mae' or args.scorer == 'both'):
            gpr_MRI_mae = GaussianProcessRegressor(kernel=best_kernel_MRI_mae,random_state=0).fit(X_MRI_train_val,Y_train_val)
            Y_MRI_predict,std_MRI = gpr_MRI_mae.predict(X_MRI_test,return_std=True)
            gpr_meg_mae = GaussianProcessRegressor(kernel=best_kernel_meg_mae,random_state=0).fit(X_meg_train_val,Y_train_val)
            Y_meg_predict,std_meg = gpr_meg_mae.predict(X_meg_test,return_std=True)  
            Y_MRI_meg_predict = np.stack((Y_MRI_predict,std_MRI,Y_meg_predict,std_meg),axis=1)           
            best_Y_predict = np.concatenate((best_Y_predict_MRI_mae,best_Y_predict_meg_mae),axis=1) 
            best_rf_mae = RandomForestRegressor(n_estimators=best_rf_params_mae[0],max_depth=best_rf_params_mae[1]).fit(best_Y_predict,Y_train_val)
            Y_predict_stacked = best_rf_mae.predict(Y_MRI_meg_predict)        
            scores_mae_f.append(mae(Y_test,Y_predict_stacked))            
        if (args.scorer == 'r2' or args.scorer == 'both'):
            gpr_MRI_r2 = GaussianProcessRegressor(kernel=best_kernel_MRI_r2,random_state=0).fit(X_MRI_train_val,Y_train_val)
            Y_MRI_predict,std_MRI = gpr_MRI_r2.predict(X_MRI_test,return_std=True)
            gpr_meg_r2 = GaussianProcessRegressor(kernel=best_kernel_meg_r2,random_state=0).fit(X_meg_train_val,Y_train_val)
            Y_meg_predict,std_meg = gpr_meg_r2.predict(X_meg_test,return_std=True)  
            Y_MRI_meg_predict = np.stack((Y_MRI_predict,std_MRI,Y_meg_predict,std_meg),axis=1)
            best_Y_predict = np.concatenate((best_Y_predict_MRI_r2,best_Y_predict_meg_r2),axis=1) 
            best_rf_r2 = RandomForestRegressor(n_estimators=best_rf_params_r2[0],max_depth=best_rf_params_r2[1]).fit(best_Y_predict,Y_train_val)
            Y_predict_stacked = best_rf_r2.predict(Y_MRI_meg_predict)        
            scores_r2_f.append(r2_score(Y_test,Y_predict_stacked))

        Y_pred_array.append(Y_predict_stacked+age_mean)
        Y_chron_age_array.append(Y_test+age_mean)
            
    scores = np.array([scores_mae_f,scores_r2_f])
    Y_pred_array = np.array(Y_pred_array)
    Y_chron_age_array = np.array(Y_chron_age_array)
    fname_pred = 'Age_prediction_'+scorer+'_CCA2_GPR_Stacking.txt'
    fname_chron_age = 'Age_chronological_'+scorer+'_CCA2_GPR_Stacking.txt'
    np.savetxt(fname_pred,Y_pred_array,fmt='%5.8f')
    np.savetxt(fname_chron_age,Y_chron_age_array,fmt='%5.8f')
    return scores        

if __name__=='__main__':
    
    folds = 2
    all_scores_mae = []
    all_scores_r2 = []
    
    with Pool(processes=2) as pool:
        res = list(tqdm(pool.imap(CCA_GPR_Stacking_nCV_fold, list(range(folds))),total=folds))
        
    res = np.array(res)    
    all_scores_mae = np.ravel(res[:,0,:])
    all_scores_r2 = np.ravel(res[:,1,:])

    if (args.scorer == 'mae' or args.scorer == 'both'):
        np.savetxt(filename_to_write(scorer='mae'), all_scores_mae,fmt='%5.8f')
    if (args.scorer == 'r2' or args.scorer == 'both'):
        np.savetxt(filename_to_write(scorer='r2'), all_scores_r2,fmt='%5.8f')