#!/usr/bin/env python
# coding: utf-8
import cv2

import os
import subprocess
import glob
import argparse
import numpy as np
import re
import multiprocessing
import time

import torch

import pdb
import sys
import tqdm

import distmat

modules = os.path.dirname(os.path.realpath(__file__)) + '/..'         
sys.path.insert(0, modules)

try:
    from modules.models.DALF import DALF_extractor as PGNet
except:
    raise ImportError('Unable to import model')  

def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

experiment_name = ''
exp_dir_target = ''


def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input path containing single or several (use -d flag) PNG-CSV dataset folders"
	, required=True, default = 'lists.txt') 
	parser.add_argument("-o", "--output", help="Output path where results will be saved."
	, required=True, default = '.') 
	parser.add_argument("-f", "--file", help="Use file list with several input dirs instead (make sure -i points to .txt path)"
	, action = 'store_true') 
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true')
	parser.add_argument("--sift", help="Run with keypoints instead of ground-truth .csv"
	, action = 'store_true')
	parser.add_argument("--tps_path", help="Directory with refined TPS path and keypoints"
	, required=False, default = '')
	parser.add_argument("-np", "--net_path", help="pretrained weights path for model if applicable"
    , required=False, default = '')


	args = parser.parse_args()

	if args.sift and (args.tps_path == '' or not os.path.exists(args.tps_path)):
		raise RuntimeError('--sift requires a valid --tps_path folder')

	return args


def correct_cadar_csv(csv):
	for line in csv:
		if line['x'] < 0 or line['y'] < 0:
			line['valid'] = 0


def gen_keypoints_from_csv(csv):
	keypoints = []
	for line in csv:
		if line['valid'] == 1:
			k = cv2.KeyPoint(float(line['x']), float(line['y']),15.0, 0.0) 
			k.class_id = int(line['id'])
			keypoints.append(k)

	return keypoints	 

			
def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]
	return dirs or False

def save_dist_matrix(ref_kps, ref_descriptors, ref_gt, tgt_kps, descriptors, tgt_gt, out_fname):
	#np.linalg.norm(a-b)
	print ('saving matrix in:',  out_fname)
	size = len(ref_gt)
	dist_mat = np.full((size,size),-1.0,dtype = np.float32)
	valid_m = 0
	matches=0

	matching_sum = 0

	begin = time.time()

	for m in range(len(ref_kps)):
		i = ref_kps[m].class_id
		if ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			valid_m+=1
		for n in range(len(tgt_kps)):
			j = tgt_kps[n].class_id
			if ref_gt[i]['valid'] and tgt_gt[i]['valid'] and tgt_gt[j]['valid']:
				dist_mat[i,j] = np.linalg.norm(ref_descriptors[m]-descriptors[n]) #distance.euclidean(ref_d,tgt_d) #np.linalg.norm(ref_d-tgt_d)

	print('Time to match NRLFeat: %.3f'%(time.time() - begin))

	mins = np.argmin(np.where(dist_mat >= 0, dist_mat, 65000), axis=1)
	for i,j in enumerate(mins):
		if i==j and ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			matches+=1

	print ('--- MATCHES --- %d/%d'%(matches,valid_m))

	with open(out_fname, 'w') as f:

		f.write('%d %d\n'%(size,size))

		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.8f '%(dist_mat[i,j]))


def get_gt_idx(kp, kp_gt):
	kp_dict = {}
	for idx, k in enumerate(kp):
		kp_dict['%.2f,%.2f'%(k.pt[0],k.pt[1])] = idx

	try:
		gt_idx = [kp_dict['%.2f,%.2f'%(k.pt[0],k.pt[1])] for k in kp_gt]
	except Exception as e:
		print(e)
		pdb.set_trace()

	return gt_idx

def run_benchmark(args):

	dev = torch.device('cuda' if torch.cuda.is_available else 'cpu')
	extractor = PGNet(model = args.net_path, dev = dev, fixed_tps=False)
	exp_name = 'dalf'

	print('Running benchmark')

	ref_descriptor = None
	ref_gt = None

	if args.file:
		exp_list = get_dir_list(args.input)
	elif args.dir:
		exp_list = [d for d in glob.glob(args.input+'/*/*') if os.path.isdir(d)]
	else:
		exp_list = [args.input]

	exp_list = list(filter(lambda x: 'DeSurTSampled' in x or  'Kinect1' in x or 'Kinect2Sampled' in x or 'SimulationICCV' in x, exp_list))
	#exp_list = list(filter(lambda x: 'SimulationICCV' in x , exp_list))

	for exp_dir in tqdm.tqdm(exp_list):

		dataset_name = os.path.join(*os.path.abspath(exp_dir).split('/')[-2:]) ; #print(dataset_name) ; input()

		experiment_files = glob.glob(exp_dir + "/*-rgb*")

		#print(experiment_files) ; input()
	
		master_f = ''
		for exp_file in experiment_files:
			if 'master' in exp_file or 'ref' in exp_file:
				fname = exp_file.split('-rgb')[0]
				#print(fname) ; input()
				if not args.sift:
					ref_gt = np.recfromcsv(fname + '.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					correct_cadar_csv(ref_gt)
					ref_kps = gen_keypoints_from_csv(ref_gt)
					#print(ref_descriptors.shape) ; input()
				else:
					tps_fname = os.path.join(args.tps_path, *fname.split('/')[-3:])
					ref_gt = np.recfromcsv(tps_fname + '.sift', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					ref_kps = distmat.load_cv_kps(ref_gt)
					#for kp in ref_kps:
					#	kp.size *= 0.5
					#print(tps_fname) ; input()
					
				img = fname + '-rgb.png'
				#ref_descriptors = extractor.compute(img, ref_kps)
				# __ref_kps, ref_descriptors = extractor.detectAndCompute(img)
				# gt_idx = get_gt_idx(__ref_kps, ref_kps)
				# ref_descriptors = ref_descriptors[gt_idx]
				ref_descriptors = np.load(tps_fname + '.pgnet.npy')
				distmat.save_desc(tps_fname+ '.pgdeal', ref_descriptors)
				master_f = exp_file

		for exp_file in experiment_files:

			if 'master' not in exp_file and 'ref' not in exp_file:
				fname = exp_file.split('-rgb')[0]
				if not args.sift:
					tgt_gt = np.recfromcsv(fname + '.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					correct_cadar_csv(tgt_gt)
					tgt_kps = gen_keypoints_from_csv(tgt_gt)
				else:
					tps_fname = os.path.join(args.tps_path, *fname.split('/')[-3:])
					tgt_gt = np.recfromcsv(tps_fname + '.sift', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					tgt_kps = distmat.load_cv_kps(tgt_gt)
					#for kp in tgt_kps:
					#	kp.size *= 0.5 #np.random.randint(0,360)

				img = fname + '-rgb.png'
				#descriptors = extractor.compute(img, tgt_kps)
				# __tgt_kps, descriptors = extractor.detectAndCompute(img)
				# gt_idx = get_gt_idx(__tgt_kps, tgt_kps)
				# descriptors = descriptors[gt_idx]
				descriptors = np.load(tps_fname + '.pgnet.npy')
				distmat.save_desc(tps_fname+ '.pgdeal', descriptors)

				if not args.sift:
					mat_fname = os.path.basename(master_f).split('-rgb')[0] + '__' + os.path.basename(exp_file).split('-rgb')[0] + \
								'__' + exp_name + '.txt'
				else:
					mat_fname = os.path.basename(master_f).split('-rgb')[0] + '__' + os.path.basename(exp_file).split('-rgb')[0] + \
								'__' + exp_name	+ '.dist'			

				result_dir = os.path.join(args.output,experiment_name) + '/' + dataset_name + '/' + exp_dir_target
				check_dir(result_dir)
				#ref_descriptors, ref_gt = descriptors, tgt_gt
				if not args.sift:
					save_dist_matrix(ref_kps,ref_descriptors,ref_gt, tgt_kps, descriptors,tgt_gt, os.path.join(result_dir,mat_fname))
				else:
					distmat.save(ref_descriptors, descriptors, os.path.join(result_dir,mat_fname))


run_benchmark(parseArg())
