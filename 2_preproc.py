# -*- coding: utf-8 -*-
"""
@author: Alba
"""
import os
import os.path as op
import glob
from tqdm import tqdm
import mne
from mne.report import Report

## ~~~~~~~~  PARAMETERS
rest_dir     = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest'
report_path  = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports'
logFile      = open('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\preproc_logFile.txt','a')
report       = Report()
## ~~~~~~~~  PARAMETERS

def preproc(subject):
    global report
    from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
    from mne.io import read_raw_fif
    import numpy as np
    
    raw_path = op.join(rest_dir, subject, 'meg_clean','nose-pts-out_raw.fif')
    clean_raw_path = op.join(rest_dir, subject, 'meg_clean','clean_raw.fif')
    if not os.path.exists(raw_path):                                            # skip subjects that have T1 but do not have MEG data
        return
    raw = read_raw_fif(raw_path, preload=True)

    pick_meg    = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, ref_meg=False)
    pick_eog1   = mne.pick_channels(raw.ch_names, ['EOG061'])
    pick_eog2   = mne.pick_channels(raw.ch_names, ['EOG062'])
    pick_ecg    = mne.pick_channels(raw.ch_names, ['ECG063'])
    if pick_eog1.size==0 or pick_eog2.size==0 or pick_ecg.size==0:
        logFile.write("Subject " + subject + ": EOG and/or ECG channels not found.\n")
        return        

    ##-- Filtering, PSD --##
    fig = raw.plot_psd(area_mode='range', average=False, picks=pick_meg)
    report.add_figs_to_section(fig, captions= subject + ' => PSD before filtering', section='Filtering')
    raw.notch_filter((50,100), filter_length='auto', fir_design='firwin', phase='zero', picks=np.append(pick_meg,(pick_ecg,pick_eog1,pick_eog2)))
    raw.filter(1., None, fir_design='firwin', phase='zero') # high-pass filtering at 1 Hz
    raw.resample(200, npad="auto")  # set sampling frequency to 200Hz (antialiasing filter at 100 Hz)
    
    ##-- Artifact removal --##
    random_state = 23                           # we want to have the same decomposition and the same order of components each time it is run for reproducibility
    ica = ICA(n_components=25, method='fastica', random_state=random_state)
    reject = dict(mag=5e-12, grad=4000e-13)     # avoid fitting ICA on large environmental artifacts that would dominate the variance and decomposition    
    try:
        ica.fit(raw, picks=pick_meg, reject=reject)
    except:
        logFile.write("Subject " + subject + ": ICA could not run. Large environmental artifacts.\n")
        return
    fig = ica.plot_components(inst=raw)
    report.add_figs_to_section(fig, captions=[subject + ' => ICA components']*len(fig), section='Artifact removal')
    
    # EOG artifact
    eog_epochs = create_eog_epochs(raw, reject=reject)
    if eog_epochs.events.size != 0:
        eog_inds, scores = ica.find_bads_eog(eog_epochs)
        if len(eog_inds) != 0:    
            fig = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},image_args={'sigma': 1.})
            report.add_figs_to_section(fig, captions=[subject + ' => ICA correlated with EOG']*len(fig), section='Artifact removal')
            ica.exclude.extend(eog_inds)
        else:
            logFile.write("Subject " + subject + ": No ICA component correlated with EOG\n")
    else:
        logFile.write("Subject " + subject + ": No EOG events found\n")
    
    # ECG artifact
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
    if ecg_epochs.events.size != 0:
        ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        if len(ecg_inds) != 0:    
            fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.})
            report.add_figs_to_section(fig, captions=[subject + ' => ICA correlated with ECG']*len(fig), section='Artifact removal')
            ica.exclude.extend(ecg_inds)
        else:
            logFile.write("Subject " + subject + ": No ICA component correlated with ECG\n")
    else:
        logFile.write("Subject " + subject + ": No ECG events found\n")
        
    raw_clean = raw.copy()
    ica.apply(raw_clean)    
    
    fig = raw_clean.plot_psd(area_mode='range', picks=pick_meg)
    report.add_figs_to_section(fig, captions= subject + ' => PSD after filtering and artifact correction', section='Filtering')
    
    raw_clean.save(clean_raw_path, overwrite=True)

if __name__ == '__main__':
    mne.set_log_level(verbose=False)
    
    subjects_dir = 'G:\OpenSource_data\CAM-CAN\\freesurfer_anat'
    os.chdir(subjects_dir)
    files = glob.glob('sub-CC*')
    
    for subject in tqdm(files):
        preproc(subject)
    
    report.save(op.join(report_path, 'Preproc_sub-CC7.html'), overwrite=True, open_browser=False) 
    logFile.close()