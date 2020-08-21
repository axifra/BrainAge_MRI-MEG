# -*- coding: utf-8 -*-
"""
@author: Alba
"""
import os
import os.path as op
import mne
from mne.report import Report
from mne.io import read_raw_fif
import glob
from tqdm import tqdm
import warnings

def coreg_head2mri(subjects_dir, subject, native_fid, raw_path, raw_NosePtsOut, trans_dst, flag_fid=False):
    import scipy.io
    import numpy as np
    from mne.gui._coreg_gui import CoregModel
    
    model = CoregModel()
    model.mri.subjects_dir = subjects_dir
    model.mri.subject = subject
    
    # Remove Polhemus points around the nose (y>0, z<0)
    model.hsp.file = raw_path
    head_pts = model.hsp.points
    raw  = read_raw_fif(raw_path, preload=True)
    pos  = np.where((head_pts[:,1] <= 0) | (head_pts[:,2] >= 0))                                   
    dig  = raw.info['dig']
    dig2 = dig[0:8]
    dig3 = [dig[p+7] for p in pos[0]]
    dig_yeah = dig2+dig3
    raw.info['dig'] = dig_yeah
    raw.save(raw_NosePtsOut, overwrite=True)
    
    model.hsp.file = raw_NosePtsOut
    
    # Load CamCAN fiducials from matlab file
    if flag_fid:
        fid = scipy.io.loadmat(native_fid, struct_as_record=False, squeeze_me=True)
        fid = fid['fid']
  
        model.mri.lpa = np.reshape(fid.native.mm.lpa*0.001,(1,3))
        model.mri.nasion = np.reshape(fid.native.mm.nas*0.001,(1,3))
        model.mri.rpa = np.reshape(fid.native.mm.rpa*0.001,(1,3))
        assert (model.mri.fid_ok)
        
        lpa_distance = model.lpa_distance
        nasion_distance = model.nasion_distance
        rpa_distance = model.rpa_distance
              
        model.nasion_weight = 1.
        model.fit_fiducials(0)
        old_x = lpa_distance ** 2 + rpa_distance ** 2 + nasion_distance ** 2
        new_x = (model.lpa_distance ** 2 + model.rpa_distance ** 2 +
                 model.nasion_distance ** 2)
        assert new_x < old_x
        
    avg_point_distance = np.mean(model.point_distance)
    
    while True:
        model.fit_icp(0)
        new_dist = np.mean(model.point_distance)
        assert new_dist < avg_point_distance
        if model.status_text.endswith('converged)'):
            break
    
    model.save_trans(trans_dst)
    trans = mne.read_trans(trans_dst)
    np.testing.assert_allclose(trans['trans'], model.head_mri_t)
    
if __name__ == '__main__':
    mne.set_log_level(verbose=False)
    subjects_dir = 'G:\OpenSource_data\CAM-CAN\\freesurfer_anat'
    rest_dir     = 'G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest'
    os.chdir(subjects_dir)
    files   = glob.glob('sub-CC*')
    report  = Report()
    logFile = open('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\coregistration_logFile.txt','a')
        
    for subject in tqdm(files):
        native_fid      = op.join('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\\fiducials','fid-native-' + subject.replace("sub-","") +'.mat')
        raw_path        = op.join(rest_dir, subject, 'meg\\transdef_mf2pt2_rest_raw.fif')
        clean_dir       = op.join(rest_dir, subject, 'meg_clean')
        if not os.path.exists(clean_dir):
            os.makedirs(clean_dir)
        raw_NosePtsOut  = op.join(clean_dir,'nose-pts-out_raw.fif')
        trans_dst       = op.join(clean_dir,'_trans.fif')
        
        if not os.path.exists(raw_path):
            logFile.write("Subject " + subject + " doesn't have MEG data\n")
            continue
        
        if os.path.exists(native_fid):
            flag_fid = True
            coreg_head2mri(subjects_dir, subject, native_fid, raw_path, raw_NosePtsOut, trans_dst, flag_fid)
        else:
            logFile.write("Subject " + subject + " doesn't have fiducials\n")
            coreg_head2mri(subjects_dir, subject, native_fid, raw_path, raw_NosePtsOut, trans_dst)
        
        raw = read_raw_fif(raw_NosePtsOut, preload=True)
        fig = mne.viz.plot_alignment(raw.info, trans=trans_dst, subject=subject, dig=True, meg=True, subjects_dir=subjects_dir, surfaces=['head','inner_skull'], coord_frame='meg', show_axes=True)
        report.add_figs_to_section(fig, captions=subject, section='Co-registration')
        fig = mne.viz.plot_alignment(raw.info, trans=trans_dst, subject=subject, dig=True, meg=True, subjects_dir=subjects_dir, surfaces=['head','inner_skull'], coord_frame='meg', show_axes=True)
        with warnings.catch_warnings(record=True):
            from mayavi import mlab
            mlab.view(175, 90, distance=0.6, focalpoint=(0., 0., 0.))
        report.add_figs_to_section(fig, captions=subject, section='Co-registration')
        report.save('G:\OpenSource_data\CAM-CAN\cc700\mri\pipeline\\release004\BIDSsep\megmax_rest_reports\\Coregistration_sub-CC6.html', overwrite=True, open_browser=False) 
    
    logFile.close()