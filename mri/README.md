# MRI preprocessing and feature compilation codes
Codes for MRI data processing using FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki):
* [fsl_cmds.py](fsl_cmds.py): Automating the pipeline for brain extraction, linear and non-linear warping to register to the MNI152 template brain. Implements multiprocessing to parallely process multiple subjects from the CamCan dataset. 
* [prep_struc_feats.py](prep_struc_feats.py): Prepares csv/npz files for particular features -- GM, WM, CSF, cortical only, subcortical only. Relies on command line arguments for selecting features to extract and the output file name.
* Reference files/masks provided in [standard_masks](standard_masks/) folder. _Missing: MNI152_T1_2mm_brain.nii.gz_
* [project_loadings_MNI.py](project_loadings_MNI.py): Project the CCA loadings for each feature to the MNI152 template brain space for 3D visualization.
