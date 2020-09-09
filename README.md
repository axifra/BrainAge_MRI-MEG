# 
# Estimating brain age from structural MRI and MEG data: Insights from dimensionality reduction techniques

This repository provides Python scripts for our work presented in:

A. Xifra-Porxas*, A. Ghosh*, G.D. Mitsis, M-H. Boudrias (2019). Estimating brain age from structural MRI and MEG data: Insights from dimensionality reduction techniques. bioRxiv, * equal contribution

# Overview of the repository
Codes for the MEG data analysis using the MNE toolbox (https://mne.tools/):
- [1_coreg_head2mri.py](/meg/1_coreg_head2mri.py): Automated coregistration of a subject's head shape (using the fiducials and Polhemus headpoints) with the subject's MRI. Note that since the MRI had been defaced, the Polhemus headpoints around the nose had to be removed before coregistration for a sucessful result.
- [2_preproc.py](/meg/2_preproc.py): Notch filtering, high-pass filtering, resampling, and ICA denoising of cardiac and ocular artifacts.
- [2_preproc_manual.py](/meg/2_preproc_manual.py): Simple program that goes through the subjects for whom the automatic ICA denoising did not work, and allows to manually remove the ICA components that were seen in the generated figures to contain artifacts.
- [3_source_recon_beam.py](/meg/3_source_recon_beam.py): Source reconstruction using beamforming. It also extracts the time series for the specified brain parcellations. 
- [4_feat.py](/meg/4_feat.py): Extraction of features for the specified frequency bands. The features extracted were the absolute power at each brain parcel (later transformed to relative power), amplitude envelope correlation (a measure of functional connectivity between brain parcels; Brookes et al., 2012), and interlayer coupling (Tewarie et al., 2016). 
- Note: The codes do not use parallel processing but it is recommended as they can take a long time to run. You can take a look at the codes for the predictive models for examples on how to implement parallel processing.

Codes for MRI data processing using FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki):
- [fsl_cmds.py](/mri/fsl_cmds.py): Automating the pipeline for brain extraction, linear and non-linear warping to register to the MNI152 template brain. Implements multiprocessing to parallely process multiple subjects from the CamCan dataset. 
- [prep_struc_feats.py](/mri/prep_struc_feats.py): Prepares csv/npz files for particular features -- GM, WM, CSF, cortical only, subcortical only. Relies on command line arguments for selecting features to extract and the output file name.
- Reference files/masks provided in [standard_masks](/mri/standard_masks/) folder. _Missing: MNI152_T1_2mm_brain.nii.gz_
- [project_loadings_MNI.py](/mri/project_loadings_MNI.py): Project the CCA loadings for each feature to the MNI152 template brain space for 3D visualization.

Codes for brain age prediction using scikit-learn (https://scikit-learn.org/):
- [GPR.py](/Age_Prediction/GPR.py): Fits a GPR model on raw features
- [GPR_similarity.py](/Age_Prediction/GPR_similarity.py): Fits a GPR model on the similarity metric calculated with respect to the subjects in training dataset
- [PCA_GPR.py](/Age_Prediction/PCA_GPR.py): Fits a GPR model on the (top) principal components of the feature matrix
- [CCA_GPR.py](/Age_Prediction/CCA_GPR.py): Fits a GPR model on the canonical correlation component obtained from the covariance of features and age
- [CCA_BS.py](/Age_Prediction/CA_BS.py): Retrieves the bootstrapped ratio of loadings corresponding to each feature in the CCA model

# Authors
Alba Xifra-Porxas & Arna Ghosh

McGill University, Montreal, Canada

Please do not hesitate to contact us if you have any questions related to the use of these scripts.

E-mail: axifra@gmail.com, arna.ghosh@mail.mcgill.ca

Date: 19 August 2020
