# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:45:39 2018
@author: Alba
"""
import os
import os.path as op
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import copy
import mne
from mne.io import read_raw_fif
from mne.report import Report

## ~~~~~~~~  PARAMETERS
subjects_dir = 'G:\OpenSource_data\CAM-CAN\\freesurfer_anat'
rest_dir     = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest'
## ~~~~~~~~  PARAMETERS
    
def power_feat_beam(subject,parc,freq_bands):
    Fs = 200
    #labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir) 
    ts = np.load(op.join(rest_dir,subject, 'meg_clean','ts-beam-' + parc + '-LEAK.npy'))
    #ts = ts[:-2,:]     # remove medial wall (the .npy file with leakage correction applied has already removed the medial wall)
    ts_psd = np.zeros((ts.shape[0],freq_bands.shape[0]))
    for i,fb in enumerate(freq_bands):     
        temp = mne.time_frequency.psd_array_multitaper(ts, Fs, fmin=fb[0], fmax=fb[1], bandwidth=None, adaptive=False, low_bias=True, normalization='length', n_jobs=1, verbose=None)
        ts_psd[:,i] = np.mean(temp[0],1)
        
    np.save(op.join(rest_dir, subject,'meg_clean', 'psd' + '-beam-' + parc + '-LEAK'), ts_psd)
        
def conn_feat(subject, parc,freq_bands):
    ts = np.load(op.join(rest_dir, subject,'meg_clean', 'ts-beam-' + parc + '-LEAK.npy'))
    #ts = ts[:-2,:]     # remove medial wall (the .npy file with leakage correction applied has already removed the medial wall)
    nROIs = ts.shape[0]
    n_fb = freq_bands.shape[0]
    
    # Filter time-courses, Hilbert transform to obtain the amplitude envelope
    ts_bp = np.zeros((nROIs*n_fb,ts.shape[1]))
    ts_env = np.zeros((nROIs*n_fb,ts.shape[1]))
    for i,f in enumerate(freq_bands):
        b, a = signal.butter(3, f, btype='bandpass', fs=200)        
        ts_bp[i*nROIs:(i+1)*nROIs,:] = signal.filtfilt(b,a,ts)
        analytic_signal = signal.hilbert(ts_bp[i*nROIs:(i+1)*nROIs,:])
        ts_env[i*nROIs:(i+1)*nROIs,:] = np.abs(analytic_signal)
    
    # Amplitude Envelope Correlation (within and between frequencies)
    AEC = np.corrcoef(ts_env)    
#    fig = plt.figure()
#    im = plt.imshow(AEC-np.eye(AEC.shape[0]))
#    fig.colorbar(im,orientation='vertical')   
    
    # flatten AEC
    AEC_within_vec = np.zeros((n_fb,np.int((nROIs*nROIs-nROIs)/2)))
    AEC_between_vec = np.zeros((np.int((n_fb*n_fb-n_fb)/2),np.int((nROIs*nROIs-nROIs)/2)))
    c = 0;
    for i in range(0,n_fb):
        AEC_within = AEC[i*nROIs:(i+1)*nROIs, i*nROIs:(i+1)*nROIs]
        AEC_within_vec[i,:] = AEC_within[np.triu_indices(AEC_within.shape[0], k = 1)]
        for j in range(i+1,n_fb):
            AEC_between = AEC[i*nROIs:(i+1)*nROIs, j*nROIs:(j+1)*nROIs]
            AEC_between_vec[c,:] = AEC_between[np.triu_indices(AEC_between.shape[0], k = 1)]
            c=c+1

    # Interlayer coupling 
    ILC = np.zeros((n_fb,n_fb))
    for i in range(0,n_fb):
        temp = np.arange(n_fb)
        np.delete(temp,1)
        for j in temp:
             aux = AEC[i*nROIs:(i+1)*nROIs, j*nROIs:(j+1)*nROIs]
             xcorr = np.corrcoef(AEC_within_vec[i,:], aux[np.triu_indices(len(aux), k = 1)])
             ILC[i,j] = xcorr[0,1]
             
    ILC_vec = np.concatenate((ILC[np.triu_indices(len(ILC), k = 1)],ILC[np.tril_indices(len(ILC), k = -1)]))         

    np.savez(op.join(rest_dir, subject,'meg_clean', 'conn-beam-'+ parc + '-LEAK'), AEC, AEC_within_vec, AEC_between_vec, ILC, ILC_vec)
        
        
if __name__ == '__main__':
    mne.set_log_level(verbose=False)
    
    os.chdir(rest_dir)
    files = glob.glob('sub-CC*')
    parc  = 'aparc.a2009s'
    freq_bands = np.array([[2,4],[4,8],[8,10],[10,13],[13,26],[26,35],[35,48]])

    for subject in tqdm(files):       
        try:
            power_feat_beam(subject,parc,freq_bands)
            conn_feat(subject,parc,freq_bands)
        except:  
            tqdm.write('Error subject: '+subject)
            continue        