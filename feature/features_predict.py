import numpy as np
import h5py
import gc
import extracted_features_prep as efp
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def totalCorrectPred(pred,y):
	maxIndex = pred.argmax(axis=1)
	count = 0
	for i,j in enumerate(maxIndex):
		if y[i,j] == 1: count += 1
	return count

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]


def compileFeaturesModel(nb_classes):
	f_model = Sequential()
	f_model.add(Dense(512, input_shape=(167,10)))
	f_model.add(Flatten())
	f_model.add(Dense(nb_classes, W_regularizer=l2(0.01)))
	f_model.add(Activation('softmax'))
	

	f_model.load_weights('features_stream_model.h5')

	print 'Compiling f_model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	f_model.compile(loss='hinge',optimizer=sgd, metrics=['accuracy'])
	return f_model

def f_getTrainData(chunk,nb_classes):
	X_train,Y_train = efp.stackExtractedFeatures(chunk,'train')
	if (X_train is not None and Y_train is not None):
		#X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def CNN():
	batch_size= 8
	nb_classes = 3
	nb_epoch = 10
	chunk_size=16
	f_total_predictions = 0
	f_correct_predictions = 0
	print 'Loading dictionary...'

	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_test_data=pickle.load(f1)

	f_model = compileFeaturesModel(nb_classes)

	keys = temporal_test_data.keys()

	instance_count=0
	for chunk in chunks(keys,chunk_size):
		X_batch,Y_batch=f_getTrainData(chunk,nb_classes)
		if (X_batch is not None and Y_batch is not None):
			preds = f_model.predict_proba(X_batch)
			print (preds)
			print ('-'*40)
			print (Y_batch)

			f_total_predictions += X_batch.shape[0]
			f_correct_predictions += totalCorrectPred(preds,Y_batch)

			comparisons=[]
			maximum=np.argmax(Y_batch,axis=1)
			for i,j in enumerate(maximum):
				comparisons.append(preds[i][j])
			with open('compare.txt','a') as f1:
				f1.write(str(comparisons))
				f1.write('\n\n')
		else : 
			print "features_stream_cnn.py: X or Y_batch is None"
	print "\nThe accuracy was found out to be: ",str(f_correct_predictions*100/f_total_predictions)

if __name__ == "__main__":
	CNN()
