''' 
	Required input: Matrix of distances in ASCII .txt format (2 lines):
		m n #line 1 - size of query and target descriptors 
		dist11 dist12 ... dist1n dist21 dist22 ... dist2n . . . distn1  distn2 ... distmn. #line 2 - distance matrix saved in a single line.
	- The true correspondences lie in the diagonal (i == j)
	- The file name is used to get automatic labels for plotting (template: [img query name]__[image target name]__[descriptor name].txt)  
	- Invalid correspondeces are set to -1 in the matrix and will not be considered to calculate the curve.
	
	Usage:
	python plotPrecisionRecall.py --input /home/user/exp1/ # will read all .txt files in 'exp1' dir and plot them
	or
	python plotPrecisionRecall.py --input /home/user/plot_many.txt -f # will plot for a list of datasets

	Notice that the script will generate intermediate files to generate latex tables with the metrics.
'''
from __future__ import print_function
import argparse
import glob
import os
import re
import collections
import pdb
import hashlib
import pickle
from collections import OrderedDict


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylab import rcParams
import numpy as np


rcParams['figure.figsize'] = 15, 11
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


#from numpy import trapz, simps
from scipy.integrate import trapz, simps

#global variables
experiments = {}
c_plot = 1

Verbose = False
plotDist = False # set true to show the distance matrix as an image
nbOfPRPoints = 90 # number of 1-precision x recall points to be generated for each curve
Metric = None

row_list = []
mean_list = []

mean_dict = OrderedDict()

def show_mean_table():
	table = np.loadtxt('mean.txt', dtype = np.float32)
	mean = np.mean(table, axis = 0)
	table_row = r'Mean & \multicolumn{1}{c}{$*$}'

	#print(table.T[8])

	recog = mean #mean[:int(len(mean)/2)]
	#auc = mean[int(len(mean)/2):]

	best_recog = float(int(np.max(recog) * 100))/100 
	#best_auc = float(int(np.max(auc)*100))/100

	for i in range(len(recog)):
		if recog[i] >= best_recog:
			table_row += r'& $\textbf{%.2f}$ '%(recog[i])
		else:
			table_row += r'& $%.2f$ '%(recog[i])

	'''
	table_row+=r'& '

	for i in range(len(auc)):
		if auc[i] >= best_auc:
			table_row += r'& $\textbf{%.2f}$ '%(auc[i])
		else:
			table_row += r'& $%.2f$ '%(auc[i])		
	'''
	#print(table_row+ r'\\')
	row = '& MS'
	for v in mean:
		row += ' & $%.2f$ '%(v)
	print(row)
	print(table_row)

def show_global_mean():
	datasets = glob.glob('./*.dict')
	dicts = []
	for d in datasets:
		with open(d, 'rb') as f:
			dicts.append(pickle.load(f))

	import copy
	g_mean = copy.deepcopy(dicts[0])
	for k,v in g_mean.items():
		g_mean[k] = []


	print('loaded %d dicts'%(len(dicts)))

	for i,d in enumerate(dicts):
		for k, v in d.items():
			g_mean[k] += v
		print('Dataset ' + datasets[i])
		for k,v in d.items():
			avg_ms =np.mean(np.array(v))
			print('Method: %s, score: %.2f'%(k ,avg_ms))
		print('----------------------------------------')

	for k,v in g_mean.items():
		avg_ms =np.mean(np.array(v))
		print('Method: %s, score: %.2f'%(k ,avg_ms))

	#show_mdtable()
	#show_ms_mma_table()

def show_mdtable():
	#datasets = glob.glob('./*.dict')
	datasets = ['Kinect1','Kinect2Sampled','DeSurTSampled', 'SimulationICCV']
	dicts = []
	for d in datasets:
		with open('./dicts_sift/'+d+'.dict', 'rb') as f:
			dicts.append(pickle.load(f))

	md_header1= '|Method'
	md_header2 = '|:-----'
	
	md_table = {}

	import copy
	g_mean = copy.deepcopy(dicts[0])
	for k,v in g_mean.items():
		g_mean[k] = []

	for d in datasets:
		dname = os.path.splitext(d)[0]
		md_header1+='|%s'%(dname)
		md_header2+='|-----:'

	md_header1+='|Mean|'
	md_header2+='|-----:|'

	for i,d in enumerate(dicts):
		for k, v in d.items():
			g_mean[k] += v
		#print('Dataset ' + datasets[i])
		for k,v in d.items():
			avg_ms =np.mean(np.array(v))
			#print('Method: %s, score: %.3f'%(k ,avg_ms))
			if k in md_table:
				md_table[k]+='%.2f|'%(avg_ms)
			else:
				md_table[k] = '|%.2f|'%(avg_ms)
		#print('----------------------------------------')

	for k,v in g_mean.items():
		avg_ms =np.mean(np.array(v))
		#print('Method: %s, score: %.3f'%(k ,avg_ms))
		md_table[k]+='%.2f|'%(avg_ms)

	print(md_header1)
	print(md_header2)
	for row_name, row_val in md_table.items():
		if 'tfeat-impl' in row_name:
			row_name = 'TFeat**'
		elif 'logpolar-impl' in row_name:
			row_name = 'LogPolar**'
		elif 'DEAL' in row_name:
			row_name = 'Ours'
		print('|%s'%(row_name) + row_val)


def show_ms_mma_table():
	#datasets = glob.glob('./*.dict')
	datasets = ['Kinect1','Kinect2Sampled','DeSurTSampled', 'SimulationICCV']
	dicts_ms = []
	dicts_mma = []

	idir = './dicts_otherkps2/'

	for d in datasets:
		with open(idir+d+'_MS.dict', 'rb') as f:
			dicts_ms.append(pickle.load(f))

	for d in datasets:
		with open(idir+d+'_MMA.dict', 'rb') as f:
			dicts_mma.append(pickle.load(f))

	md_header1= '|Method'
	md_header2 = '|:-----'
	
	md_table = {}

	import copy
	g_mean_ms = copy.deepcopy(dicts_ms[0])
	g_mean_mma =  copy.deepcopy(dicts_mma[0])

	for k,v in g_mean_ms.items():
		g_mean_ms[k] = []

	for k,v in g_mean_mma.items():
		g_mean_mma[k] = []

	for d in datasets:
		dname = os.path.splitext(d)[0]
		md_header1+='|%s'%(dname)
		md_header2+='|-----:'

	md_header1+='|Mean|'
	md_header2+='|-----:|'

	for i in range(len(dicts_ms)):
		d_ms = dicts_ms[i]
		d_mma = dicts_mma[i]

		for k, v in d_ms.items():
			g_mean_ms[k] += v
			g_mean_mma[k] += d_mma[k]

		#print('Dataset ' + datasets[i])
		for k,v in d_ms.items():
			avg_ms = np.mean(np.array(v))
			avg_mma = np.mean(np.array(d_mma[k]))
			#print('Method: %s, score: %.3f'%(k ,avg_ms))
			if k in md_table:
				md_table[k]+='%.2f / %.2f|'%(avg_ms, avg_mma)
			else:
				md_table[k] = '|%.2f / %.2f|'%(avg_ms, avg_mma)
		#print('----------------------------------------')

	for k,v in g_mean_ms.items():
		avg_ms =np.mean(np.array(v))
		avg_mma = np.mean(np.array(g_mean_mma[k]))
		#print('Method: %s, score: %.3f'%(k ,avg_ms))
		md_table[k]+='%.2f / %.2f|'%(avg_ms, avg_mma)

	print(md_header1)
	print(md_header2)
	for row_name, row_val in md_table.items():
		if 'tfeat-impl' in row_name:
			row_name = 'TFeat**'
		elif 'logpolar-impl' in row_name:
			row_name = 'LogPolar**'
		elif 'DEAL' in row_name:
			row_name = 'Ours'
		print('|%s'%(row_name) + row_val)


def read_matches(filepath):
	csv = np.recfromcsv(filepath + '.match', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
	try:
		len(csv['idx_ref'])
		len(csv['idx_tgt'])
		src = csv['idx_ref']
		tgt = csv['idx_tgt']
	except:
#		print(csv['idx_ref'])
		src = np.array([csv['idx_ref']])
		tgt = np.array([csv['idx_tgt']])
		# print(src)
		# print(tgt) ; input()

	return src, tgt

def generatePRPointVectorized(Mats,threshold): #Vectorized operaton speeds up a lot
			
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	Precision = 1.0
	Recall = 0.0
	
	for M in Mats: # for each pair in the experiments (ref,tgt_i)

		N = np.array(M['mat'])
		diag1 = np.diag(N)[:M['K']].copy()

		#remove main sub-diagonal for FP computing
		N[np.diag_indices(M['K'])] = 9999
		
		TP += float(np.sum(diag1 < threshold))
		FP += float(np.sum(N < threshold))
				
		FN += float(np.sum(diag1 >= threshold))
	
	if TP + FP > 0.0:
		Precision = TP / (TP + FP)
		
	Recall = TP / (TP + FN)
	
	#print TP, FP, FN
	return (1.0 - Precision, Recall)


def generatePRPoint(Mats,threshold): # We are going to generate a single point of 1 - Precision vs Recall according to a threshold
			
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	Precision = 1.0
	Recall = 0.0
	
	for M in Mats: # for each pair in the experiments (ref,tgt_i)
		rows, cols = M.shape

		for i in range(rows):
			for j in range(cols):
				if M[i,j] >= 0: #If it is a valid match
					if M[i,j] < threshold: #It classified as a match
						if i == j: #It is correct
							TP+=1.0
						else: #It is wrong
							FP+=1.0
					else: #It classified as a mismatch
						if i == j: #It is wrong
							FN+=1.0
						else:
							TN+=1.0 #It is correct
	
	if TP + FP > 0.0:
		Precision = TP / (TP + FP)
		
	Recall = TP / (TP + FN)
	
	#print TP, FP, FN
	return (1.0 - Precision, Recall)


def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input directory containing distance matrices for all experiments or a .txt file"
	, required=True, default = '.') 
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true') 
	parser.add_argument("-mean", "--mean", help="show mean file instead of plotting (used for main table in the paper)"
	, action = 'store_true')
	parser.add_argument("-gmean", "--gmean", help="show global mean(used for ablations and parameter analysis)"
	, action = 'store_true')
	parser.add_argument("--tps_path", help="Directory with refined TPS path and keypoints"
	, required=True, default = '')	
	parser.add_argument("-m", "--mode", help="Mean Mode (append or erase mean)"
	, required=True, choices = ['append', 'erase']) 
	parser.add_argument("--metric", help="Metric to use"
	, required=True, choices = ['MS', 'MMA', 'inliers']) 
	args = parser.parse_args()
	global Metric ; Metric = args.metric
	return args


def check_consistency():
	return
	global experiments

	desc_names = list(experiments.keys())
	targets = list(experiments[desc_names[0]].keys())
	n = len(targets)

	for dname in desc_names:
		if len(experiments[dname]) != n:
			print('ERROR: Different number of mats!')
			print(dname)
			print(desc_names[0])
			print('pred %d expected %d'%(len(experiments[dname]), n))
			quit()


	for t in targets:
		for d in desc_names:
			if experiments[d][t]['mat'].shape != experiments[desc_names[0]][t]['mat'].shape:
				print('Error, different nb of mat keypoints!')
				print(t,d,targets[0])
				quit()

	'''for target in targets:
		nones = (experiments[desc_names[0]][target] == -1).sum()
		for dname in desc_names:
			if (experiments[dname][target] == -1).sum() != nones:
				#print(dname, target)
				print('Warning: Different number of invalid points in same targets! ({0}, {1})'.format(dname, target))
				#quit()
	'''
	
def getAccuracy(mat, K = None):
	#return 0
	rows,cols = mat.shape
	if K is None:
		K = rows

	if rows == 0 or cols == 0:
		return 0
	#print mat.shape
	hits=0
	total = 0
	mins = np.argmin(np.where(mat >= 0, mat, 65000), axis=1)
	#print mins ; raw_input()
	for i in range(K):
		if i == mins[i]:
			hits+=1
		if np.any(mat[i,:]>=0) or np.any(mat[:,i] >= 0):
			total+=1

	if total==0: return 0 #pdb.set_trace()	
	
	if Metric == "MS": #Matching Score
		return float(hits)/float(min(rows,cols))
	elif Metric == "MMA": #Accuracy
		return float(hits)/float(total)
	else:
		return float(hits)

def readDistMatrix(abs_filename, tps_path): # read all experiments from a dataset inside a folder
	filename, _ = os.path.splitext(os.path.basename(abs_filename))
	name_src, name_tgt, name_desc = re.split('__',filename)
	#if 'OURS' in name_desc:
	#	return
	#if 'geopatch' in name_desc:#name_desc == 'geopatchHOG':#'DAISY':
	#	return
	#if 'ORB' in name_desc:
	#	return
	#if 'FREAK' in name_desc:
	#	return
	#if 'SIFT' in name_desc:
	#	return

	if name_desc == 'OURS':
		name_desc = 'GeoBit'
	if name_desc == 'LOGPOLAR':
		name_desc = 'Log-Polar'
	if name_desc =='GEOPATCHCNN':
		name_desc = 'GeoPatch'
	if name_desc =='geopatchBRIEF':
		name_desc = 'GeoBit'
	if name_desc == 'GEOPATCHDDEPTH':
		name_desc = 'GeoPatch-D'
	if name_desc == 'GEOPATCHDDEPTHREF':
		name_desc = 'GeoPatch-DR'
	if name_desc == 'nrlfeat-geopatch-corrected-4epochs_960':
		name_desc = 'NRLFeat'
	if name_desc == 'D2NetOpenCV':
		name_desc = 'D2Net'
	if name_desc == 'newdata-DEAL-big_newdata-DEAL-big':
		name_desc = 'DEAL'
	if name_desc == 'logpolar-impl-margin0d5_logpolar-impl-margin0d5':
		name_desc = 'Log-Polar**'
	if name_desc == 'tfeat-impl_tfeat-impl':
		name_desc = 'TFeat**'

	  
	# dict_tps =  {
	# 			'LFNet':	'/srv/storage/datasets/cadar/nonrigid/neurips_rebuttal/lf-net-release/gt_tps',
	# 			'SOSNet': 	'/srv/storage/datasets/cadar/nonrigid/neurips_rebuttal/SOSNet/gt_tps',
	# 			'D2Net': 	'/srv/storage/datasets/cadar/nonrigid/neurips_rebuttal/d2-net/gt_tps',
	# 			'LIFT': 	'/srv/storage/datasets/cadar/nonrigid/neurips_rebuttal/lift/gt_tps',
	# 			'SuperPoint': '/srv/storage/datasets/cadar/nonrigid/neurips_rebuttal/SuperGluePretrainedNetwork/gt_tps',
	# 			'R2D2': 	'/srv/storage/datasets/nonrigiddataset/NEURIPS21/r2d2_tps',
	# 			'ASLFeat':	 '/srv/storage/datasets/nonrigiddataset/NEURIPS21/aslfeat_tps'
	# 			}

	# tps_path = dict_tps[name_desc]
	
	global experiments, c_plot

	matches_path = os.path.join(tps_path, *os.path.dirname(abs_filename).split('/')[-2:], name_tgt)
	idx_src, idx_tgt = read_matches(matches_path)



	#print(matches_path); input()
	#pdb.set_trace()
	
	with open(abs_filename) as f:
		print('Loading ', abs_filename)
		n_src,n_tgt = list(map(int,f.readline().rstrip('\n').split(' '))) #read length of keypoints 1 and 2		
		lin_matrix = list(map(float,f.readline().rstrip('\n').split(' ')[:-1]))
		
		dist_mat = np.array(lin_matrix, dtype=np.float32)
		dist_mat = dist_mat.reshape(n_src, n_tgt)
		dist_mat[np.where(np.isnan(dist_mat))] = -1

		#mins = np.argmin(dist_mat, axis = 1)

		'''
		#check the accuracy by index
		cnt = 0
		for i, m in enumerate(idx_src):
			if mins[m] == idx_tgt[i]:
				cnt+=1
		print('Accuracy: ', cnt / len(idx_src))
		print('Match score ', cnt / min(n_src, n_tgt))
		'''
		#Order matrix by Ground-Truth

		#if len(idx_src) == 0 or len(idx_tgt) == 0 or n_src == 0:
		#	pdb.set_trace()

		all_idx = set(range(n_src))
		src_set = set(idx_src)
		# first, sort by src keypoints


		ordered_distmat_src = np.zeros_like(dist_mat)
		ordered_distmat_src[:len(idx_src), :] = dist_mat[idx_src, :]
		ordered_distmat_src[len(idx_src):, :] = dist_mat[list(all_idx - src_set), :]

		all_idx = set(range(n_tgt))
		tgt_set = set(idx_tgt)
		# now sort by tgt keypoints
		ordered_distmat = np.zeros_like(dist_mat)
		ordered_distmat[:, :len(idx_tgt)] = ordered_distmat_src[:, idx_tgt]
		ordered_distmat[:, len(idx_tgt):] = ordered_distmat_src[:, list(all_idx - tgt_set)]

			#import traceback
			#traceback.print_exc()
			#pdb.post_mortem()
								
		if name_desc not in experiments:
			experiments[name_desc] = {}
			
		experiments[name_desc][name_tgt] = {'mat': ordered_distmat, 'K': len(idx_src)}
		
		print('[Matrix of Distances] Loaded ', filename, 
		'// Matching Score:', '%.4f'%(getAccuracy(ordered_distmat, len(idx_src))) )#, '... (%d,%d %d - %d)' % (n_src,n_tgt, k, n_src*n_tgt)
		#input()
		if plotDist:
			plt.subplot(2,3,c_plot) ; c_plot+=1
			plt.imshow(dist_mat,cmap='rainbow')
			plt.colorbar()
			plt.title(name_desc)		

			

def set_plotparams():
	matplotlib.rcParams.update({'font.size': 12})
	plt.xlabel('1 - Precision', {'fontsize':16, 'weight':'bold'}, labelpad = 5)
	plt.ylabel('Recall',{'fontsize':16, 'weight':'bold'}, labelpad = 26)
	
def plt_show(exp_path, name, use_legend = True):
	#plt.legend(loc='best')#(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   #ncol=2, mode="expand", borderaxespad=0.)

	if use_legend:
		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

	plt.grid(linestyle='--')

	#plt.show()
	print("saving plot in '%s'"%(os.path.abspath(exp_path)))
	
	if len(exp_path) > 2:
		plt.savefig(exp_path, bbox_inches='tight')
		
	plt.savefig(name+'.pdf', bbox_inches='tight')
	#print name + '.png'

def normalize_mats(mats):
	lmax = []
	
	for mat in mats:
		lmax.append(np.amax(mat))

	#pdb.set_trace()
		
	amax = np.amax(lmax)

	if np.isnan(amax): pdb.set_trace()
			
	for i in range(len(mats)):
		mats[i] = (mats[i])/(amax)
		
'''
def normalize_mats(mats):
	lmax = []
	
	for mat in mats:
		lmax.append(np.amax(mat))
		
	amax = np.amax(lmax)
			
	for i in range(len(mats)):
		mats[i] = (mats[i])/(amax)
'''		

def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]

	return dirs


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_grouped_bar_chart(acc, out_path):
	# set width of bar
	barWidth = 0.3#0.25
	colors_dict = {'SIFT':'hotpink','GeoPatch':'r','GeoBit-Heatflow':'blueviolet','Log-Polar':'g','BRAND':'gray','DaLI':'b','TFeat':'aqua', 'ORB':'black', 'FREAK':'tomato', 'DAISY':'greenyellow', 'GeoBit':'gold', 'geopatchHOG':'tomato', 'geopatchConcat':'pink','geopatchIntensity':'indigo', 'GeoPatch-D':'tomato', 'GeoPatch-DR':'gold', 'NRLFeat':'indigo'}#, 'gold', 'tomato']
	print('Saving bar plot in: ', out_path)
	bars = {}
	datasets = []
	rx = []
	#print(acc) ; input()
	#acc  = collections.OrderedDict(sorted(acc.items()))
	pos = 0
	
	for dataset, descs in acc.items():

		datasets.append(dataset)

		for desc_name, val in list(acc.values())[0].items():
			bars[desc_name] = []

		n_descs = len(bars)

		for desc_name, val in descs.items():
			bars[desc_name].append(val)
		
		sorted_accs = []
		for desc_name, bar in bars.items():
			sorted_accs.append((bar, desc_name))

		sorted_accs.sort(key = lambda x: x[0], reverse = True)
		my_cmap = plt.cm.get_cmap('jet', 32)

		i=0
		for bar, desc_name in sorted_accs:
			'''if i==0:
				rx.append( np.arange(0.5,len(bar)*2 +0.5,2) )
				pdb.set_trace()
			else:
				rx.append([x + barWidth for x in rx[i-1]])
			'''
			#print(get_cmap(200)(100)) ; input()
			#vhash = int(hashlib.sha256(desc_name[:4].encode()).hexdigest(),16)
			#color = get_cmap(64)(vhash%64)
			if desc_name not in colors_dict:
				colors_dict[desc_name] = my_cmap(np.random.randint(32))
			plt.bar(pos + i*barWidth, bar, color=colors_dict[desc_name], width=barWidth, edgecolor='white', label=desc_name)
			i+=1
		pos += barWidth * (n_descs + 2)

	# Add xticks on the middle of the group bars
	plt.xlabel('Dataset', fontweight='bold', fontsize = 17)
	plt.ylabel('Avg. Matching Score', fontweight='bold', fontsize = 17)
	plt.xticks([barWidth * (n_descs + 2)*r + (n_descs / 2) * barWidth for r in np.arange(len(datasets))], datasets, fontsize=15, rotation =70)
	plt.yticks(fontsize=16)
	#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = collections.OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04,1), loc="upper left")	

	plt.grid(linestyle='--', axis = 'y')
		
	plt.savefig(out_path+'.png', bbox_inches='tight')
	plt.savefig(out_path+'.pdf', bbox_inches='tight')


def plot_accuracies(acc):

	objects = []
	vals = []

	for name, val in acc.iteritems():
		objects.append(name)
		vals.append(val)

	y_pos = np.arange(len(objects))
	 
	plt.bar(y_pos, vals, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Mean Accuracy')
	#plt.title('Programming language usage')

def main():
	
	curves = {}
	final_acc = {}
	
	global experiments
	erased = False
	
	args = parseArg()

	if args.mean:
		show_mean_table() ; quit()
	if args.gmean:
		show_global_mean() ; quit()

	if args.dir:
		paths = [d for d in glob.glob(args.input+'/*') if os.path.isdir(d)]
	else:
		paths = [args.input]

	#pdb.set_trace()
	
	#print(args.input)
	#print(glob.glob(args.input+'/../*')) ; input()

	for dataset in paths:
		if 'chambre_light' in dataset or 'chambre_medium' in dataset \
			or 'cloth' in dataset or 'stone' in dataset \
			or 'Bag1' in dataset \
			or 'newspaper2' in dataset:
			continue

		dataset = dataset.rstrip('/')
		experiments = {}
		curves = {}
		accuracies = {}
		
		dataset_name =  os.path.basename(dataset) #os.path.abspath(dataset).split('/')[-3]
		#print(dataset) ; input()
		
		experiment_files = glob.glob(dataset + '/cloud_master__*.dist')#"/*.txt")
		
		#Load the matrices of 'responses' for each pairwise match for all descriptors
		for f in experiment_files:
			readDistMatrix(os.path.abspath(f), args.tps_path)
		check_consistency() # Check the consistency of the loaded mats

		if plotDist:
			plt.show()

		dataset_size = len(list(experiments.values())[0].values())
		
		for desc_name, pairwise_exps in experiments.items(): #for all descriptors, calculate 1-precision x recall
			
			print('[Plot] Generating curve for', desc_name, '...')
			
			Mats = []
			curve = []	
			
			for exp_name, table in pairwise_exps.items():			
				Mats.append(table)
			
			#normalize_mats(Mats) #Normalize distances

			accuracies[desc_name] = np.mean([getAccuracy(m['mat'], m['K']) for m in Mats])

			try:
				max_dist = 0.
				for m in Mats:
					max_dist = max(max_dist, np.max(m['mat']))
			except:
				max_dist = 1e-6
				
			#thresholds = np.concatenate((np.linspace(0.0, 0.1, nbOfPRPoints*0.05) , np.linspace(0.1, 0.6, nbOfPRPoints*0.9) , np.linspace(0.6, 0.9, nbOfPRPoints*0.05)))
			#varying threshold 
			for threshold in np.linspace(0., max_dist, nbOfPRPoints):
				PRPoint = [1.0, 1.0] #generatePRPointVectorized(Mats,threshold)
				curve.append(PRPoint)
				if Verbose:
					print('[Plot]', 'threshold', threshold, PRPoint)

			curves[desc_name] = curve
			
		markers = ['*','v','^','s','o']	
		linestyles = ['-', '--', '-.', ':']
		colors = ['r','blueviolet','g','chocolate','b','dimgrey','fuchsia']
		#m_pos=0
		
		all_plots = []

		for plot_name, curve in curves.items():	#plot all curves
			
			x = [PR[0] for PR in curve]
			y = [PR[1] for PR in curve]	

			auc = trapz(y,x)#trapz(y,x)

			all_plots.append((auc, plot_name ,x, y))

		all_plots.sort(key = lambda x: x[0], reverse= True)

		for auc, plot_name, x, y in all_plots:
			m_pos = abs(hash(plot_name))
			leg_name = plot_name + ' (auc: %.3f)' % (auc)
			plt.plot(x,y,marker=markers[m_pos%len(markers)], linestyle=linestyles[m_pos%len(linestyles)], color= colors[m_pos%len(colors)] ,label=leg_name,lw=2)			
			
		
		set_plotparams()
		
		plot_name = ''
		name_chunks= os.path.abspath(dataset).split('/')[-3:]
		for chunk in name_chunks:
			plot_name += chunk + '--'
		plot_name = plot_name[:-2]
		
		print(plot_name) ; 
	
		#plt_show(dataset + '/' + os.path.basename(os.path.abspath(dataset+'/'))+'_plot.png', plot_name+'_prc')
		#plt.close()

		final_acc[dataset_name] = accuracies

		all_plots.sort(key = lambda x: x[1] if 'DEAL' not in x[1] else 'Z'+x[1])


		header = ''
		all_accs = []
		all_aucs = []
		for auc,plot_name, _, _ in all_plots:
			header+= r'& ' + plot_name
			all_accs.append(accuracies[plot_name])
			all_aucs.append(auc)
		print(header + '\n')

		mean_row = ''
		table_row = r'& \multicolumn{1}{l}{%s ($%d$)}'%(dataset_name.replace(r'_',r'\_'), dataset_size)
		best_acc = sorted(all_accs)[-2:]
		best_auc = sorted(all_aucs)[-2:]

		out_acc = ''
		
		print('--------------')
		best_acc = [np.round(x,2) for x in best_acc]
		for auc, plot_name, x, y in all_plots:
			if np.round(accuracies[plot_name],2) == best_acc[0]:
				table_row += r'& $\underline{%.2f}$ '%(accuracies[plot_name])
			elif np.round(accuracies[plot_name],2) == best_acc[1]:
				table_row += r'& $\textbf{%.2f}$ '%(accuracies[plot_name])
			else:
				table_row += r'& $%.2f$ '%(accuracies[plot_name])
			mean_row+=str(accuracies[plot_name]) + ' '
			print('Method: %s, Matching Score: %.3f'%(plot_name, accuracies[plot_name]))
			if plot_name in mean_dict:
				mean_dict[plot_name].append(accuracies[plot_name])
			else:
				mean_dict[plot_name] = [accuracies[plot_name]]
		print('--------------')
			#print plot_name ; raw_input()
		
		'''
		table_row+=r'& '

		for auc, plot_name, x, y in all_plots:
			if auc == best_auc:
				table_row += r'& $\textbf{%.2f}$ '%(auc)
			else:
				table_row += r'& $%.2f$ '%(auc)
			mean_row+=str(auc) + ' '
		'''

		row_list.append(table_row + r' \\')
		mean_list.append(mean_row)

			#print '%s %.2f %.2f' % (plot_name, auc, accuracies[plot_name]) ; raw_input()
		
		#plt_show(dataset + '/' + os.path.basename(os.path.abspath(dataset+'/'))+'_plot_acc.png', plot_name + '_acc', False)
		#plt.close()
	#plot_grouped_bar_chart(final_acc, out_path=args.input + '/' + 'bar_plotAcc')

	for row in row_list:
		print(row)

	mode = 'w'

	if args.mode == 'erase' and not erased:
		mode = 'w'
		erased = True
	else: 
		mode = 'a+'

	with open('mean.txt', mode) as f:
		for m in mean_list:
			f.write(m + '\n')

	with open('table.txt', mode) as f:
		for row in row_list:
			f.write(row + '\n')

		f.write('\n')

	#& AUC & $0.20$ & $0.16$ & $\mathit{0.31}$ & $0.22$ & $0.28$ & $0.25$ & $0.26$ & $\mathit{0.31}$ & $\mathbf{0.32}$  \\

	curr_dataset = os.path.basename(args.input)

	avg_row_str = '& MS '
	print('\n------- average MS --------')
	avg_ms_arr = np.array([np.mean(np.array(v)) for k,v in mean_dict.items()])
	max_ms = np.round(np.max(avg_ms_arr),2)

	for k, v in mean_dict.items():
		avg_ms =np.mean(np.array(v))
		print('Method: %s, score: %.3f'%(k ,avg_ms))
		avg_ms = np.round(avg_ms, 2)
		# if avg_ms == max_ms:
		# 	avg_row_str+=' & $\\mathit{%.2f}$ '%(avg_ms)
		# else:
		avg_row_str+=' & $%.2f$ '%(avg_ms)
	print('-----------------------------')

	print(header)
	print(avg_row_str)

	with open('table_means.txt', mode) as f:
		f.write(header+'\n')
		f.write(avg_row_str+'\n')

	with open(curr_dataset+ '_' + args.metric +'.dict', 'wb') as f:
		pickle.dump(mean_dict, f, pickle.HIGHEST_PROTOCOL)


main()
