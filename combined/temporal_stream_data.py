import numpy as np
import sys,os
import pickle
import optical_flow_prep as ofp
import gc


def stackOF(chunk,img_rows,img_cols,jobType):
	if jobType == 'train':
		pickleFile = '../dataset/temporal_train_data.pickle'
	else :
		pickleFile = '../dataset/temporal_test_data.pickle'
	with open(pickleFile,'rb') as f1:
		temporal_train_data=pickle.load(f1)

	X_train,Y_train=ofp.stackOpticalFlow(chunk,temporal_train_data,img_rows,img_cols)
	gc.collect()
	return (X_train,Y_train)
