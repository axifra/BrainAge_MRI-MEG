import numpy as np
import nibabel as nib
import os, subprocess
from tqdm import tqdm
from multiprocessing import Pool

ref_folder = "/usr/share/fsl/data/standard/"

MNI_template = nib.load(os.path.join(ref_folder,'MNI152_T1_2mm_brain.nii.gz'))
MNI_template_data = MNI_template.get_data()
orig_size = MNI_template_data.shape
print orig_size
#MNI_template_reshaped = MNI_template_data.reshape(np.prod(MNI_template_data.shape),)
MNI_loadings = np.zeros(orig_size)
MNI_loadings = MNI_loadings.reshape(np.prod(MNI_loadings.shape),)

# add WM corrected loadings

wm_all_loadings = np.genfromtxt('Data/WM_BS_loadings.csv', delimiter=',', dtype=np.float64)
wm_corrected_loadings = wm_all_loadings[2,:]

wm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_WM.nii.gz'
wm_mask_object = nib.load(wm_mask_path)
wm_mask_data = wm_mask_object.get_data()
wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

MNI_loadings[wm_mask_data_full_bool] = wm_corrected_loadings

# add GM corrected loadings

gm_all_loadings = np.genfromtxt('Data/GM_BS_loadings.csv', delimiter=',', dtype=np.float64)
gm_corrected_loadings = gm_all_loadings[2,:]

gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
gm_mask_object = nib.load(gm_mask_path)
gm_mask_data = gm_mask_object.get_data()
gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

MNI_loadings[gm_mask_data_full_bool] = gm_corrected_loadings


MNI_loadings = MNI_loadings.reshape(orig_size)
print(MNI_loadings.min(),MNI_loadings.max(),MNI_loadings.shape)

MNI_loadings_image = nib.Nifti1Image(MNI_loadings, MNI_template.affine, MNI_template.header)

nib.save(MNI_loadings_image, os.path.join('Data','GM+WM_MNI_loadings.nii.gz'))
