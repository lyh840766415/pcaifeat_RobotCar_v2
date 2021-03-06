import pickle
import numpy as np
import os
import cv2
import random


BASE_PATH = "/data/lyh/benchmark_datasets"

def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories

def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries

def get_pc_img_match_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("point image match Loaded.")
		return queries	

def load_pc_file(filename):
	filename = os.path.join(BASE_PATH,filename)
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,3))
		exit()
		
	#returns Nx3 matrix
	#print(filename)
	pc=np.fromfile(filename, dtype=np.float64)
	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])
		print(filename)
		exit()

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_pc_files(filenames):
	pcs=[]
	for filename in filenames:
		#print(filename)
		pc,success=load_pc_file(filename)
		if not success:
			return np.array([]),False
		#if(pc.shape[0]!=4096):
		#	continue
		pcs.append(pc)
	pcs=np.array(pcs)
	return pcs,True

def load_pc_file_save_2_txt(filename,output_dir):
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,3)),True
		
	#returns Nx3 matrix
	#print(filename)
	pc=np.fromfile(filename, dtype=np.float64)
	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([]),True

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	np.savetxt(os.path.join(output_dir,"%s.xyz"%(filename[-20:-4])), pc, fmt="%.5f", delimiter = ',')
	return pc,True
	
def load_image(filename):
	filename = os.path.join(BASE_PATH,filename)
	#return scaled image
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((288,144,3))
		
	img = cv2.imread(filename)
	img = cv2.resize(img,(288,144))
	return img

def load_images(filenames):
	imgs=[]
	for filename in filenames:
		#print(filename)
		img,success=load_image(filename)
		if not success:
			return np.array([]),False
		imgs.append(img)
	imgs=np.array(imgs)
	return imgs,True

def load_img_pc(load_pc_filenames,load_img_filenames,pool):
	pcs = []
	imgs = []
	if load_pc_filenames != None:
		pcs = pool.map(load_pc_file,load_pc_filenames)
		pcs = np.array(pcs)
	if load_img_filenames != None:
		imgs = pool.map(load_image,load_img_filenames)
		imgs=np.array(imgs)
	
	return pcs,imgs
	