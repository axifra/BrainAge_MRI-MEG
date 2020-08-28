import numpy as np
import nibabel as nib
import os, subprocess
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import pandas as pd

def cortical_intensities_Destrieux_parcels(subject):

	# get Destrieux mask
	Destrieux_mask_path = 'Data/Destrieux_2mm_cropped.nii.gz'
	Destrieux_mask_object = nib.load(Destrieux_mask_path)
	Destrieux_mask_data = Destrieux_mask_object.get_data()
	Destrieux_parcel_labels = np.unique(Destrieux_mask_data)	# 0th label is 0 (black pixels, not important)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()

	parcel_intensity_feats = []

	for parcel in range(1,len(Destrieux_parcel_labels)):
		final_mask = Destrieux_mask_data==Destrieux_parcel_labels[parcel]
	
		parcel_intensities = sub_image_data[final_mask]
		parcel_intensity_feats.append(np.array([np.median(parcel_intensities),np.std(parcel_intensities)]))
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	parcel_intensity_feats = np.array(parcel_intensity_feats)
	return parcel_intensity_feats

def cortical_intensities_GM(subject):

	# get cortical mask
	cortical_mask_path = 'Data/MNI152_T1_2mm_strucseg.nii.gz'
	cortical_mask_object = nib.load(cortical_mask_path)
	cortical_mask_data = cortical_mask_object.get_data()
	cortical_mask_data_reshaped = cortical_mask_data.reshape(np.prod(cortical_mask_data.shape),)
	cortical_mask_data_full_bool = cortical_mask_data_reshaped==1

	# get gm mask
	gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
	gm_mask_object = nib.load(gm_mask_path)
	gm_mask_data = gm_mask_object.get_data()
	gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
	gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

	final_mask = gm_mask_data_full_bool & cortical_mask_data_full_bool
	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_gm_masked = sub_image_reshaped[final_mask]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return sub_image_gm_masked

def subcortical_intensities_GM(subject):

	# get subcortical mask
	subcortical_mask_path = 'Data/MNI152_T1_2mm_strucseg.nii.gz'
	subcortical_mask_object = nib.load(subcortical_mask_path)
	subcortical_mask_data = subcortical_mask_object.get_data()
	subcortical_mask_data_reshaped = subcortical_mask_data.reshape(np.prod(subcortical_mask_data.shape),)
	subcortical_mask_data_full_bool = subcortical_mask_data_reshaped>1

	# get gm mask
	gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
	gm_mask_object = nib.load(gm_mask_path)
	gm_mask_data = gm_mask_object.get_data()
	gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
	gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

	final_mask = gm_mask_data_full_bool & subcortical_mask_data_full_bool
	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_gm_masked = sub_image_reshaped[final_mask]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return sub_image_gm_masked


def all_feats(subject):

	# get wm mask
	wm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_WM.nii.gz'
	wm_mask_object = nib.load(wm_mask_path)
	wm_mask_data = wm_mask_object.get_data()
	wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
	wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

	# get gm mask
	gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
	gm_mask_object = nib.load(gm_mask_path)
	gm_mask_data = gm_mask_object.get_data()
	gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
	gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

	# get csf mask
	csf_mask_path = 'Data/MNI152_T1_2mm_brain_seg_CSF.nii.gz'
	csf_mask_object = nib.load(csf_mask_path)
	csf_mask_data = csf_mask_object.get_data()
	csf_mask_data_reshaped = csf_mask_data.reshape(np.prod(csf_mask_data.shape),)
	csf_mask_data_full_bool = csf_mask_data_reshaped.astype(bool)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_wm_masked = sub_image_reshaped[wm_mask_data_full_bool]
	sub_image_gm_masked = sub_image_reshaped[gm_mask_data_full_bool]
	sub_image_csf_masked = sub_image_reshaped[csf_mask_data_full_bool]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return np.concatenate((sub_image_gm_masked, sub_image_wm_masked, sub_image_csf_masked))

def both_feats(subject):

	# get wm mask
	wm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_WM.nii.gz'
	wm_mask_object = nib.load(wm_mask_path)
	wm_mask_data = wm_mask_object.get_data()
	wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
	wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

	# get gm mask
	gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
	gm_mask_object = nib.load(gm_mask_path)
	gm_mask_data = gm_mask_object.get_data()
	gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
	gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_wm_masked = sub_image_reshaped[wm_mask_data_full_bool]
	sub_image_gm_masked = sub_image_reshaped[gm_mask_data_full_bool]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return np.concatenate((sub_image_gm_masked, sub_image_wm_masked))


def only_gm(subject):

	# get gm mask
	gm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_GM.nii.gz'
	gm_mask_object = nib.load(gm_mask_path)
	gm_mask_data = gm_mask_object.get_data()
	gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
	gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_gm_masked = sub_image_reshaped[gm_mask_data_full_bool]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return sub_image_gm_masked


def only_wm(subject):

	# get wm mask
	wm_mask_path = 'Data/MNI152_T1_2mm_brain_seg_WM.nii.gz'
	wm_mask_object = nib.load(wm_mask_path)
	wm_mask_data = wm_mask_object.get_data()
	wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
	wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_wm_masked = sub_image_reshaped[wm_mask_data_full_bool]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_wm_masked)))
	return sub_image_wm_masked

def only_csf(subject):

	# get wm mask
	csf_mask_path = 'Data/MNI152_T1_2mm_brain_seg_CSF.nii.gz'
	csf_mask_object = nib.load(csf_mask_path)
	csf_mask_data = csf_mask_object.get_data()
	csf_mask_data_reshaped = csf_mask_data.reshape(np.prod(csf_mask_data.shape),)
	csf_mask_data_full_bool = csf_mask_data_reshaped.astype(bool)

	# get subject T1 image
	subject_name = "sub-"+subject
	if os.path.exists(os.path.join('Data','anat',subject_name))==False:
		tqdm.write(os.path.join('Data','anat',subject_name)+" doesn't exist")
		return None

	sub_image = nib.load(os.path.join('Data','anat',subject_name,'anat',subject_name+'_T1w_MNI.nii.gz'))
	sub_image_data = sub_image.get_data()
	sub_image_reshaped = sub_image_data.reshape(np.prod(sub_image_data.shape),)
	sub_image_csf_masked = sub_image_reshaped[csf_mask_data_full_bool]
	#tqdm.write(str(np.shape(sub_image_gm_masked))+","+str(np.shape(sub_image_csf_masked)))
	return sub_image_csf_masked


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', help="csf, gm, wm, both (gm+wm), all (gm+wm+csf), parcel_cort, cort (cortical gm only) or subcort (subcortical gm only)", default='both')
	parser.add_argument('--outFile', help="name of output file")

	args = parser.parse_args()
	print(args.mode)

	info_csv = pd.read_csv('Data/participant_data.tsv',sep='\t')
	subjects = info_csv['Observations']
	age = info_csv['age']
	subject_info = zip(subjects,age)
	sub_feats = []
	sub_ages = []
	for subject,sub_age in tqdm(subject_info):
		if args.mode=='both':
			sub_feats.append(both_feats(subject))
		elif args.mode=='all':
			sub_feats.append(all_feats(subject))
		elif args.mode=='gm':
			sub_feats.append(only_gm(subject))
		elif args.mode=='csf':
			sub_feats.append(only_csf(subject))
		elif args.mode=='cort':
			sub_feats.append(cortical_intensities_GM(subject))
		elif args.mode=='parcel_cort':
			sub_feats.append(cortical_intensities_Destrieux_parcels(subject))
		elif args.mode=='subcort':
			sub_feats.append(subcortical_intensities_GM(subject))
		else:
			sub_feats.append(only_wm(subject))
		sub_ages.append(sub_age)

	sub_feats = np.array(sub_feats)
	sub_ages = np.array(sub_ages)
	print(np.shape(sub_feats),np.shape(sub_ages))
	# sub_feats = np.hstack((sub_feats,sub_ages.reshape(np.prod(sub_ages.shape),1)))

	print(np.shape(sub_feats))
	if args.outFile==None:
		print("Please enter output file to save feature matrix")
	else:
		# np.savetxt(os.path.join('Data',args.outFile+'.csv'),sub_feats,delimiter=",")
		# np.save(os.path.join('Data',args.outFile+'.npy'),sub_feats)
		np.savez(os.path.join('Data',args.outFile+'.npz'),x=sub_feats,y=sub_ages)	# load the npz file using A = np.load(file) and then do feats = A['x'], age = A['y']
