import numpy as np
import h5py
import gc
import temporal_stream_data as tsd
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization


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

def get_activations(t_model, layer, X_batch):
	get_activations = theano.function([t_model.layers[0].input],
	t_model.layers[layer].get_output(train=False),
	allow_input_downcast=True)
	activations = get_activations(X_batch)
	return activations

def t_getTrainData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOF(chunk,img_rows,img_cols,'test')
	if (X_train is not None and Y_train is not None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def compileTemporalModel(img_channels,img_rows,img_cols,nb_classes):
	
	print 'Preparing architecture...'

	t_model = Sequential()

	t_model.add(Convolution2D(48, 7, 7, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
	t_model.add(BatchNormalization())
	t_model.add(Activation('relu'))
	t_model.add(MaxPooling2D(pool_size=(2, 2)))

	t_model.add(Convolution2D(96, 5, 5, border_mode='same'))
	t_model.add(BatchNormalization())
	t_model.add(Activation('relu'))
	t_model.add(MaxPooling2D(pool_size=(2, 2)))

	t_model.add(Convolution2D(256, 3, 3, border_mode='same'))
	t_model.add(BatchNormalization())
	t_model.add(Activation('relu'))

	t_model.add(Convolution2D(512, 3, 3, border_mode='same'))	
	t_model.add(BatchNormalization())
	t_model.add(Activation('relu'))

	t_model.add(Convolution2D(512, 3, 3, border_mode='same'))
	t_model.add(BatchNormalization())
	t_model.add(Activation('relu'))
	t_model.add(MaxPooling2D(pool_size=(2, 2)))

	t_model.add(Flatten())
	t_model.add(Dense(512))
	t_model.add(Activation('relu'))
	t_model.add(Dropout(0.7))
	t_model.add(Dense(512))
	t_model.add(Activation('relu'))
	t_model.add(Dropout(0.8))

	t_model.add(Dense(nb_classes))
	t_model.add(Activation('softmax'))

	t_model.load_weights('temporal_stream_model.h5')

	print 'Compiling t_model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	t_model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
	return t_model


def CNN():
	input_frames=10
	batch_size=8
	nb_classes = 101
	nb_epoch = 10
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=8
	t_total_predictions = 0
	t_correct_predictions = 0
	print 'Loading dictionary...'

	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_test_data=pickle.load(f1)

	t_model = compileTemporalModel(img_channels,img_rows,img_cols,nb_classes)

	keys=temporal_test_data.keys()
	random.shuffle(keys)

	for chunk in chunks(keys,chunk_size):
		X_batch,Y_batch=t_getTrainData(chunk,nb_classes,img_rows,img_cols)
		if (X_batch is not None and Y_batch is not None):
				preds = t_model.predict_proba(X_batch)
				print (preds)
				print ('-'*40)
				print (Y_batch)
				# Calcualting the total predictions and the correct ones.
				t_total_predictions += X_batch.shape[0]
				t_correct_predictions += totalCorrectPred(preds,Y_batch)

				comparisons=[]
				maximum=np.argmax(Y_batch,axis=1)
				for i,j in enumerate(maximum):
					comparisons.append(preds[i][j])
				with open('compare.txt','a') as f1:
					f1.write(str(comparisons))
					f1.write('\n\n')
		else : 
			print "X_batch or Y_batch found to be none."
	print "\nThe accuracy was found out to be: ",str(t_correct_predictions*100/t_total_predictions)

if __name__ == "__main__":
	CNN()
