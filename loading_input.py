import pickle
import numpy as np
import os
import cv2
import random


BASE_PATH = "/"

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
	return pc,True
		
	#print("pointcloud shape ", pc_4096.shape)
	#np.savetxt('result.txt', pc_4096, fmt="%.5f", delimiter = ',')
	#exit()
		#return np.array([])

	return pc_4096,True

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

def load_image(filename):
	#return scaled image
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((288,144,3)),True
		
	img = cv2.imread(filename)
	img = cv2.resize(img,(288,144))
	return img,True

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

