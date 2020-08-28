import subprocess, os
from tqdm import tqdm
import glob
from multiprocessing import Pool

def T1_MNI_register(higher_sub_dir):
	subject = higher_sub_dir[10:]
	sub_dir = os.path.join(higher_sub_dir,'anat')
	# Brain extraction pipeline
	str_bet = 'bet '+os.path.join(sub_dir,subject)+'_T1w.nii.gz '+os.path.join(sub_dir,subject)+'_T1w_brain'
	proc = subprocess.Popen(str_bet, shell=True, stdout=subprocess.PIPE)
	proc.wait()
	if proc.returncode!=0:
		tqdm.write("Error occured in BET construction for subject "+subject)
		return

	ref_folder = "/usr/share/fsl/data/standard/"
	# Linear Tfo to MNI152 space
	str_flirt = 'flirt -ref '+ref_folder+'MNI152_T1_2mm_brain.nii.gz -in '+os.path.join(sub_dir,subject)+'_T1w_brain.nii.gz -omat '+sub_dir+'affine_guess_struct2mni.mat'
	proc = subprocess.Popen(str_flirt, shell=True, stdout=subprocess.PIPE)
	proc.wait()
	if proc.returncode!=0:
		tqdm.write("Error occured in FLIRT construction for subject "+subject)
		return

	# Non-linear warping to MNI152 space
	str_fnirt = 'fnirt --ref='+ref_folder+'MNI152_T1_2mm_brain.nii.gz --in='+os.path.join(sub_dir,subject)+'_T1w_brain.nii.gz --iout='+os.path.join(sub_dir,subject)+'_T1w_brain_MNI --aff='+sub_dir+'affine_guess_struct2mni.mat --refmask='+ref_folder+'MNI152_T1_2mm_brain_mask_dil.nii.gz'
	proc = subprocess.Popen(str_fnirt, shell=True, stdout=subprocess.PIPE)
	proc.wait()
	if proc.returncode!=0:
		tqdm.write("Error occured in FNIRT construction for subject "+subject)
		return


if __name__=='__main__':
	#T1_MNI_register("Data/anat/sub-CC110033") # simple run for one subject without tqdm or pool
	
	files = glob.glob("Data/anat/sub-CC1*") # change here for which group of subjects you want to run
	files.sort()
	'''
	for file in tqdm(files[0:1]):
		T1_MNI_register(file)
	'''
	p = Pool(processes=16)
	for _ in tqdm(p.imap(T1_MNI_register,files),total = len(files)):
		pass
	p.close()
	p.join()
