# -*- coding: utf-8 -*-
"""
@author: Alba
"""

import os
import os.path as op
import mne
from mne.report import Report
from IPython import get_ipython

## ~~~~~~~~  PARAMETERS
rest_dir     = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest'
report_path  = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports'
logFile      = open('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\preproc_manual_logFile.txt','a')
report       = Report()
## ~~~~~~~~  PARAMETERS

def preproc_manual(subject):
    
    global report
    from mne.preprocessing import ICA
    from mne.io import read_raw_fif
    
    get_ipython().run_line_magic('matplotlib', 'inline')                      
    
    raw_path    = op.join(rest_dir, subject, 'meg_clean','clean_raw.fif')  
    raw = read_raw_fif(raw_path, preload=True)
    
    pick_meg    = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, ref_meg=False)
    
    fig = raw.plot_psd(area_mode='range', average=False, picks=pick_meg)
    report.add_figs_to_section(fig, captions= subject + ' => PSD before artifact correction', section='Filtering')

    ##-- Artifact removal --##
    random_state = 23                           # we want to have the same decomposition and the same order of components as in the first preprocessing
    n_components = 25
    ica = ICA(n_components, method='fastica', random_state=random_state)
    reject = dict(mag=5e-12, grad=4000e-13)     # avoid fitting ICA on crazy environmental artifacts that would dominate the variance and decomposition    
    ica.fit(raw, picks=pick_meg, reject=reject)
    fig = ica.plot_components(inst=raw)
    report.add_figs_to_section(fig, captions=[subject + ' => ICA components']*len(fig), section='Artifact removal')
    
    flag = True
    comp_inds = []
    while flag==True:
        aux = int(input('ICA components to remove? (100 to finish)'))
        if aux == 100:
            flag = False
        else:
            comp_inds.append(aux)
    
    ica.exclude.extend(comp_inds)
    str1 = ','.join(str(c) for c in comp_inds)
    logFile.write(subject + ": " + str1 + "\n")
    
    raw_clean = raw.copy()
    ica.apply(raw_clean)   
    
    fig = raw_clean.plot_psd(area_mode='range', picks=pick_meg)
    report.add_figs_to_section(fig, captions= subject + ' => PSD after artifact correction', section='Filtering')
    
    raw_clean.save(raw_path, overwrite=True)
    raw.save(op.join(rest_dir, subject, 'meg_clean','firstclean_raw.fif'), overwrite=True)


if __name__ == '__main__':
    mne.set_log_level(verbose=False)
    
    file = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\preproc_logFile.txt'
    with open(file) as f: 
          data = f.readlines() 
    
    for line in data:
        subject = line[8:20]
        if 'CC7' in subject:
            print(line)
            clean_YN = input('CLEAN SUBJECT? (Y/N)')
            if clean_YN == 'Y':
                preproc_manual(subject)
            else:
                print('Going to next subject...')
                continue

    report.save(op.join(report_path, 'Preproc_sub-CC7_manual.html'), overwrite=True, open_browser=False) 
    logFile.close()