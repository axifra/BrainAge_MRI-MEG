import numpy as np
import sys
import time
import argparse
import os
from scipy import stats
from tqdm import tqdm
import itertools
import pdb

def load_data(gm,wm,cortical,subcortical,mri,psd,aec,ilc,meg):
	X_total = None
	Y_total = None
	load_str = None
	if gm and not mri:
		data = np.load('X_GM.npy')
		X = stats.zscore(data[:,0:data.shape[1]-1])
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = data[:,data.shape[1]-1]
		if load_str is None:
			load_str = 'GM'
		else:
			load_str += ' + GM'

	if wm and not mri:
		data = np.load('X_WM.npy')
		X = stats.zscore(data[:,0:data.shape[1]-1])
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = data[:,data.shape[1]-1]
		if load_str is None:
			load_str = 'WM'
		else:
			load_str += ' + WM'

	if cortical and not mri:
		data = np.load('X_MRI_cort.npy')
		X = stats.zscore(data[:,0:data.shape[1]-1])
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = data[:,data.shape[1]-1]
		if load_str is None:
			load_str = 'cortical GM'
		else:
			load_str += ' + cortical GM'

	if subcortical and not mri:
		data = np.load('X_MRI_subcort.npy')
		X = stats.zscore(data[:,0:data.shape[1]-1])
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = data[:,data.shape[1]-1]
		if load_str is None:
			load_str = 'subcortical GM'
		else:
			load_str += ' + subcortical GM'	

	if mri:
		data = np.load('X_MRI.npy')
		X = stats.zscore(data[:,0:data.shape[1]-1])
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = data[:,data.shape[1]-1]
		if load_str is None:
			load_str = 'MRI'
		else:
			load_str += ' + MRI'

	if psd or meg:
		data = np.load('X_psd_rel_LEAK.npy')
		X = stats.zscore(data)
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = np.load('age.npy')
		if load_str is None:
			load_str = 'PSD'
		else:
			load_str += ' + PSD'

	if aec or meg:
		data = np.load('X_AEC_within_LEAK.npy')
		X = stats.zscore(data)
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = np.load('age.npy')
		if load_str is None:
			load_str = 'AEC'
		else:
			load_str += ' + AEC'

	if ilc or meg:
		data = np.load('X_ILC_LEAK.npy')
		X = stats.zscore(data)
		if X_total is None:
			X_total = X
		else:
			X_total = np.concatenate((X_total,X),axis=1)
		if Y_total is None:
			Y_total = np.load('age.npy')
		if load_str is None:
			load_str = 'ILC'
		else:
			load_str += ' + ILC'

	print('Data loading done (' + load_str +') ' + str(X_total.shape))
	return X_total, Y_total

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
	args = parser.parse_args()
	X_total, Y_total = load_data(gm=args.gm,wm=args.wm,cortical=args.cort,subcortical=args.subcort,mri=args.mri,psd=args.psd,aec=args.aec,ilc=args.ilc,meg=args.meg)
