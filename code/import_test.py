from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

datalabel="single"

def data_label():
    return datalabel

def MCNN_data_load(DATA_TYPE,GPT,protien_name):
    if GPT : #有GPT
        path_pos_train="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/new_pos/train_gpt.npy"
        path_pos_test="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/new_pos/test_gpt.npy"
        path_neg_train="../dataset/"+str(DATA_TYPE)+"/people_all/train.npy"
        path_neg_test="../dataset/"+str(DATA_TYPE)+"/people_all/test.npy"
    else :
        path_pos_train="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/pos_15/train.npy"
        path_pos_test="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/pos_15/test.npy"
        path_neg_train="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/neg_15/train.npy"
        path_neg_test="../dataset/"+str(DATA_TYPE)+"/"+str(protien_name)+"/neg_15/train.npy"
    #"../dataset/"+str(DATA_TYPE)+"/training_data/pos/data.npy"
    #"../dataset/"+str(DATA_TYPE)+"/test/pos/pos_test.npy"

    
    x_train,y_train=data_load(path_pos_train,path_neg_train)
    x_test,y_test=data_load(path_pos_test,path_neg_test)
    return(x_train,y_train,x_test,y_test)

def data_load(pos,neg):
    pos_file=np.load(pos)
    neg_file=np.load(neg)
    
    pos_label = np.ones(pos_file.shape[0])
    neg_label = np.zeros(neg_file.shape[0])
    
    x=np.concatenate([pos_file,neg_file], axis=0)
    y=np.concatenate([pos_label, neg_label], axis=0)
    y= tf.keras.utils.to_categorical(y,2)
    #y.dtype='float16'
    gc.collect()
    return x ,y