import numpy as np
import cv2 

def save(desc_ref, desc_tgt, filename):
	desc_ref = np.array(desc_ref)
	desc_tgt = np.array(desc_tgt)

	dist_mat = np.linalg.norm(desc_ref[:, None, :] - desc_tgt[None, :, :], axis=-1)

	#dist_mat = np.sqrt( 2. * (1. - desc_ref @ desc_tgt.T))


	'''
	dist_mat2 = np.zeros((desc_ref.shape[0], desc_tgt.shape[0]), dtype = np.float32)
	for i in range(dist_mat2.shape[0]):
		for j in range(dist_mat2.shape[1]):
			dist_mat2[i,j] = cv2.norm(desc_ref[i] - desc_tgt[j])
	
	print(np.linalg.norm(dist_mat - dist_mat2, ord = 'fro')) ; input()
	
	'''
	#print(desc_ref[np.isnan(desc_ref)].shape)
	#print(desc_tgt[np.isnan(desc_tgt)].shape) 
	#print(dist_mat[np.isnan(dist_mat)].shape) ; input()

	with open(filename, 'w') as f:
		f.write('%d %d\n'%(dist_mat.shape[0], dist_mat.shape[1]))
		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.5f '%(dist_mat[i,j]))
			
		f.write('\n')

def save_cvnorm(desc_ref, desc_tgt, filename):
	desc_ref = np.array(desc_ref)
	desc_tgt = np.array(desc_tgt)

	dist_mat = np.zeros((desc_ref.shape[0], desc_tgt.shape[0]), dtype = np.float32)

	for i in range(dist_mat.shape[0]):
		for j in range(dist_mat.shape[1]):
			dist_mat[i,j] = cv2.norm(desc_ref[i] - desc_tgt[j])

	with open(filename, 'w') as f:
		f.write('%d %d\n'%(dist_mat.shape[0], dist_mat.shape[1]))
		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.5f '%(dist_mat[i,j]))
			
		f.write('\n')	

def load_cv_kps(csv):
	keypoints = []
	for line in csv:
		k = cv2.KeyPoint(line['x'], line['y'], line['size']*1., line['angle'])
		keypoints.append(k)

	return keypoints

def save_desc(filename, desc):
	m, n = desc.shape
	with open(filename, 'w') as f:
		f.write('%d %d\n'%(m, n))
		for i in range(m):
			for j in range(n):
				f.write('%.8f'%(desc[i,j]))
				if j < n-1:
					f.write(',')
			f.write('\n')