# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:27:38 2019

@author: Alba
"""

import os
import os.path as op
import glob
from tqdm import tqdm
import numpy as np
import surfer
from surfer import Brain
import mne
from mne.io import read_raw_fif
from mne.report import Report

## ~~~~~~~~  PARAMETERS
subjects_dir = 'G:\OpenSource_data\CAM-CAN\\freesurfer_anat'
rest_dir     = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest'
report_path  = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports'
logFile      = open('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\source_beam_logFile.txt','a')
report       = Report()
## ~~~~~~~~  PARAMETERS

def source_recon(subject):
    global report
    from mayavi import mlab
    
    clean_path = op.join(rest_dir, subject, 'meg_clean','clean_raw.fif')
    empty_room_path = op.join('G:\OpenSource_data\CAM-CAN\cc700\meg\pipeline\\release004\emptyroom', subject[4:],'emptyroom_' + subject[4:] +'.fif')
    if not os.path.exists(clean_path):                                            
        return
    
    raw_clean = read_raw_fif(clean_path, preload=True)
    raw_empty_room = read_raw_fif(empty_room_path, preload=True)
    trans     = mne.read_trans(op.join(rest_dir, subject, 'meg_clean','_trans.fif'))

    pick_meg = mne.pick_types(raw_clean.info, meg=True, eeg=False, stim=False, ref_meg=False)

    ## Noise covariance
    raw_empty_room.filter(1., 200, fir_design='firwin', phase='zero')
    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None, picks=pick_meg)
    data_cov = mne.compute_raw_covariance(raw_clean, picks=pick_meg)
    
    ## Surface mesh 
    src     = mne.read_source_spaces(op.join(rest_dir, subject, 'meg_clean','_oct6-src.fif'))   
    
    ## Forward solution   
    fwd     = mne.read_forward_solution(op.join(rest_dir, subject, 'meg_clean','_fwd.fif'))
    
    ## Inverse model
    method  = 'beam'
    filters     = mne.beamformer.make_lcmv(raw_clean.info, fwd, data_cov, noise_cov=noise_cov)
    stc     = mne.beamformer.apply_lcmv_raw(raw_clean, filters)    
    if (len(src[0]['vertno']) != len(stc.vertices[0])) | (len(src[1]['vertno']) != len(stc.vertices[1])):
        logFile.write("Subject " + subject + ": Number of vertices of src and stc does not match, src[0] = " + str(len(src[0]['vertno'])) + ", src[1] = " + str(len(src[1]['vertno'])) + ", stc[0] = " + str(len(stc.vertices[0])) + ", stc[1] = " + str(len(stc.vertices[1])) +".\n")
        vertno = [src[0]['vertno'],src[1]['vertno']]
        stc.expand(vertno)
    stc.save(op.join(rest_dir, subject, 'meg_clean','_' + method + '_inv'))
    _, peak_time = stc.get_peak(hemi='lh')
    brain = stc.plot(initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir)
    report.add_figs_to_section(brain._figures[0], captions= subject + ' => stc plot', section='Source reconstruction')
    
    ## Extract time series from cortical parcellations
    parcellations    = ['Yeo_7Network','Yeo_17Network', 'aparc.a2009s']      #  parcellations to use, e.g., 'aparc' 'aparc.a2009s'
    for parc in parcellations:
        brain   = Brain(subject, 'lh', 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation(parc)
        report.add_figs_to_section(brain._figures[0], captions= subject + ' => ' + parc, section='Parcellations')    
        
        labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)         # Get labels from FreeSurfer's cortical parcellation
        try:
            label_ts = mne.extract_label_time_course([stc], labels_parc, src, mode='mean_flip',allow_empty=False, return_generator=False)
            label_ts = np.squeeze(np.asarray(label_ts))     # convert list to numpy array
            np.save(op.join(rest_dir, subject,'meg_clean', 'ts' + '-' + method + '-' + parc + '.npy'), label_ts)        
        except:
            logFile.write("Subject " + subject + " : Parcellation " + parc + " failed.\n")
    
    
if __name__ == '__main__':
    mne.set_log_level(verbose='ERROR')

    os.chdir(subjects_dir)
    files = glob.glob('sub-CC*') 
    
    for subject in tqdm(files):
        source_recon(subject)

    report.save(op.join(report_path, 'Source_recon_beam_sub-CC3.html'), overwrite=True, open_browser=False) 
    logFile.close()