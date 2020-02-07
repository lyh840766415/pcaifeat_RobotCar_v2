import tensorflow as tf
import numpy as np
from loading_input import *

# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 3
TRAIN_ALL = True
ONLY_TRAIN_FUSION = True

# is rand init 
RAND_INIT = False

# model path
MODEL_PATH = ""
PC_MODEL_PATH = ""
IMG_MODEL_PATH = ""

# log path
LOG_PATH = ""

# Epoch & Batch size &FINAL EMBBED SIZE
EPOCH = 20
LOAD_BATCH_SIZE = 20
FEAT_BARCH_SIZE = 2
LOAD_FEAT_RATIO = LOAD_BATCH_SIZE//FEAT_BARCH_SIZE
EMBBED_SIZE = 256

#pos num,neg num,other neg num,all_num
POS_NUM = 2
NEG_NUM = 2
OTH_NUM = 2
BATCH_DATA_SIZE = 1 + POS_NUM + NEG_NUM + OTH_NUM

# Hard example mining start
HARD_MINING_START = 5

# Margin
MARGIN1 = 0.5
MARGIN2 = 0.2

#Train file index
TRAIN_FILE = 'generate_queries/training_queries_RobotCar.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)

def init_imgnetwork():
	img_placeholder = tf.placeholder(tf.float32,shape=[BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY),144,288,3])
	endpoints,body_prefix = resnet.endpoints(images_placeholder,is_training=True)
	return img_placeholder,endpoints['model_output']
	
def init_pcnetwork(batch):
	pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY),4096,3])
	is_training_pl = tf.Variable(True, name = 'is_training')
	bn_decay = get_bn_decay(batch)
	endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)
	return pc_placeholder,endpoints
	
def init_fusion_network(pc_feat,img_feat):
	img_pc_concat_feat = tf.concat((pc_feat,img_feat),axis=1)
	pcai_feat = tf.layers.dense(img_pc_concat_feat,EMBBED_SIZE)
	return pcai_feat

def init_pcainetwork():
	#training step
	step = tf.Variable(0)
	
	#init sub-network
	if TRAINING_MODE != 2:
		pc_placeholder, pc_feat = init_pcnetwork()
	if TRAINING_MODE != 1:
		img_placeholder, img_feat = init_imgnetwork()
	if TRAINING_MODE == 3:
		pcai_feat = init_fusion_network(pc_feat,img_feat)
	
	#prepare data and loss
	if TRAINING_MODE != 2:
		pc_feat = tf.reshape(pc_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,pc_feat.shape[1]])
		q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec = tf.split(pc_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		pc_loss = quadruplet_loss(q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec, MARGIN1)
		tf.summary.scalar('pc_loss', pc_loss)
		
	if TRAINING_MODE != 1:
		img_feat = tf.reshape(img_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,img_feat.shape[1]])
		q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec = tf.split(img_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		img_loss = quadruplet_loss(q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec, MARGIN1)
		tf.summary.scalar('img_loss', img_loss)
		
	if TRAINING_MODE == 3:
		pcai_feat = tf.reshape(pcai_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,pcai_feat.shape[1]])
		q_vec, pos_vec, neg_vec, oth_vec = tf.split(pcai_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		all_loss = quadruplet_loss(q_vec, pos_vec, neg_vec, oth_vec, MARGIN1)
		tf.summary.scalar('all_loss', all_loss)
		
	#learning rate strategy, all in one?
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	#variable update
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	#only the fusion_variable
	#TODO
	fusion_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	if ONLY_TRAIN_FUSION and TRAINING_MODE == 3:
		with tf.control_dependencies(fusion_ops):
			fusion_train_op = optimizer.minimize(all_loss, global_step=step)
	
	#training operation
	with tf.control_dependencies(update_ops):
		if TRAINING_MODE != 2:
			pc_train_op = optimizer.minimize(pc_loss, global_step=step)
		if TRAINING_MODE != 1:
			img_train_op = optimizer.minimize(img_loss, global_step=step)
		if TRAINING_MODE == 3:
			all_train_op = optimizer.minimize(all_loss, global_step=step)
	
	#merged all log variable
	merged = tf.summary.merge_all()
	
	#output of pcainetwork init
	if TRAINING_MODE == 1:
		ops = {
			"pc_placeholder":pc_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"pc_train_op":pc_train_op,
			"merged":merged,
			"step":step}
		return
		
	if TRAINING_MODE == 2:
		ops = {
			"img_placeholder":img_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"img_loss":img_loss,
			"img_train_op":img_train_op,
			"merged":merged,
			"step":step}
		return ops
		
	if TRAINING_MODE == 3 and ONLY_TRAIN_FUSION:
		ops = {
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"img_loss":img_loss,
			"all_loss":all_loss,
			"pc_train_op":pc_train_op,
			"img_train_op":img_train_op,
			"all_train_op":all_train_op,
			"fusion_train_op":fusion_train_op,
			"merged":merged,
			"step":step}
		return ops
		
	if TRAINING_MODE == 3:
		ops = {
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"img_loss":img_loss,
			"all_loss":all_loss,
			"pc_train_op":pc_train_op,
			"img_train_op":img_train_op,
			"all_train_op":all_train_op,
			"merged":merged,
			"step":step}
		}
		return ops
		

def init_network_variable(sess,train_saver):
	sess.run(tf.global_variables_initializer())
	
	if RAND_INIT:
		return
		
	if TRAINING_MODE == 1:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		return
		
	if TRAINING_MODE == 2:
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		return
	
	if TRAINING_MODE == 3 and ONLY_TRAIN_FUSION:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		return
	
	if TRAINING_MODE == 3:
		train_saver['all_saver'].restore(sess,MODEL_PATH)
		return

def init_train_saver():
	all_saver = tf.train.Saver()
	variables = tf.contrib.framework.get_variables_to_restore()
	pc_variable = [v for v in variables if v.name.split('/')[0]!='pc_var']
	img_variable = [v for v in variables if v.name.split('/')[0]!='img_var']
	pc_saver = tf.train.Saver(pc_variable)
	img_saver = tf.train.Saver(img_variable)
	
	train_saver = {
		'all_saver':all_saver,
		'pc_saver':pc_saver,
		'img_saver':img_saver}
	
	return train_saver
	
def prepare_batch_data(pc_data,img_data,feat_batch,ops):
	if TRAINING_MODE != 2:
		feat_batch_pc = pc_data[feat_batch*BATCH_DATA_SIZE*FEAT_BARCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BARCH_SIZE]
	if TRAINING_MODE != 1:
		feat_batch_img = img_data[feat_batch*BATCH_DATA_SIZE*FEAT_BARCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BARCH_SIZE]
	
	if TRAINING_MODE == 1:
		train_feed_dict = {
			ops["pc_placeholder"]:feat_batch_pc,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 2:
		train_feed_dict = {
			ops["img_placeholder"]:feat_batch_img,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 3:
		train_feed_dict = {
			ops["img_placeholder"]:feat_batch_img,
			ops["pc_placeholder"]:feat_batch_pc,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(ops,train_feed_dict):
	if TRAINING_MODE == 1:
		summary,step,pc_loss,_,= sess.run([ops["merged"],ops["step"],ops["pc_loss"],ops["pc_train_op"]],feed_dict = train_feed_dict)
		print("batch num = %d , pc_loss = %f"%(step, pc_loss))

	if TRAINING_MODE == 2:
		summary,step,img_loss,_,= sess.run([ops["merged"],ops["step"],ops["img_loss"],ops["img_train_op"]],feed_dict = train_feed_dict)
		print("batch num = %d , img_loss = %f"%(step, img_loss))
	
	if TRAINING_MODE == 3:
		if ONLY_TRAIN_FUSION:
			summary,step,all_loss,_,= sess.run([ops["merged"],ops["step"],ops["all_loss"],ops["fusion_train_op"]],feed_dict = train_feed_dict)
			print("batch num = %d , all_loss = %f"%(step, all_loss))
		
		else:
			summary,step,all_loss,_,= sess.run([ops["merged"],ops["step"],ops["all_loss"],ops["all_train_op"]],feed_dict = train_feed_dict)
			print("batch num = %d , all_loss = %f"%(step, all_loss))
			
	#other training strategy
	train_writer.add_summary(summary, step)
	return step
	
def evaluate():
	return
	
def model_save(step):
	if TRAINING_MODE == 1:
		save_path = train_saver['pc_saver'].save(sess,os.path.join(LOG_PATH, "pc_model_%08d.ckpt"%(step)))
		print("Model saved in file: %s" % save_path)
		return
		
	if TRAINING_MODE == 2:
		save_path = train_saver['img_saver'].save(sess,os.path.join(LOG_PATH, "img_model_%08d.ckpt"%(step)))
		print("Model saved in file: %s" % save_path)
		return
	
	if TRAINING_MODE == 3:
		save_path = train_saver['all_saver'].save(sess,os.path.join(LOG_PATH, "model_%08d.ckpt"%(step)))
		print("Model saved in file: %s" % save_path)
		return
			
			
def main():
	#init network pipeline
	ops = init_pcainetwork()
	
	#init train saver
	train_saver = init_train_saver()

	#init GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	
	#init tensorflow Session
	with tf.Session(config=config) as sess:
		#init all the variable
		init_network_variable(sess,train_saver)
		train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
		
		#start training
		for ep in range(EPOCH):
			train_file_num = len(TRAINING_QUERIES.keys())
			train_file_idxs = np.arange(0,len(TRAINING_QUERIES.keys()))
			np.random.shuffle(train_file_idxs)
			print('train_file_num = %f , BATCH_NUM_QUERIES = %f , iteration per batch = %f' %(len(train_file_idxs), BATCH_NUM_QUERIES,len(train_file_idxs)//BATCH_NUM_QUERIES))
			
			#for each load batch
			for load_batch in range(train_file_num//LOAD_BATCH_SIZE):
				load_batch_keys = train_file_idxs[load_batch*LOAD_BATCH_SIZE:(load_batch+1)*LOAD_BATCH_SIZE]
				
				#select load_batch tuple	
				load_pc_filenames,load_img_filenames = get_load_batch_filename(load_batch_keys)
				
				#load pc&img data from file
				pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames)
				
				#for each feat_batch
				for feat_batch in range(LOAD_FEAT_RATIO):
					#prepare this batch data
					train_feed_dict = prepare_batch_data(pc_data,img_data,feat_batch,ops)
										
					#training
					step = train_one_step(ops,train_feed_dict)
					
					#evaluate TODO
					if step%201 == 0:
						evaluate()
					
					if step%3001 == 0:
						model_save(step)
					
					#TODO: Add hard mining
					'''
					if(ep > HARD_MINING_START and i%701 == 0):
            #update cached feature vectors
            TRAINING_LATENT_VECTORS=get_latent_vectors(sess, ops, TRAINING_QUERIES)
            print("Updated cached feature vectors")
          '''
						
					
if __name__ == "__main__":
	main()