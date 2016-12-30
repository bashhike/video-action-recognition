import numpy as np
import h5py
import gc
import temporal_stream_data as tsd
import extracted_features_prep as efp
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Merge

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def totalCorrectPred(pred,y):
	maxIndex = pred.argmax(axis=1)
	count = 0
	for i,j in enumerate(maxIndex):
		if y[i,j] == 1: count += 1
	return count

def f_getTrainData(chunk,nb_classes,requiredLines):
	X_train,Y_train = efp.stackExtractedFeatures(chunk,'test',requiredLines)
	if (X_train is not None and Y_train is not None):
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def t_getTrainData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOF(chunk,img_rows,img_cols,'test')
	if (X_train is not None and Y_train is not None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def prepareFeaturesModel(nb_classes,requiredLines):
	print "Preparing architecture of feature model..."
	f_model = Sequential()
	f_model.add(Dense(512, input_shape=(167,requiredLines)))
	f_model.add(Flatten())
	f_model.add(Dense(nb_classes, W_regularizer=l2(0.1)))
	f_model.add(Activation('linear'))
	f_model.add(Activation('softmax'))
	return f_model

def prepareTemporalModel(img_channels,img_rows,img_cols,nb_classes):
	
	print 'Preparing architecture of temporal model...'

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

	return t_model


def CNN():
	input_frames=10
	batch_size=10
	nb_classes = 3
	nb_epoch = 10
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=8
	requiredLines = 1000
	total_predictions = 0
	correct_predictions = 0

	print 'Loading dictionary...'
	with open('../dataset/temporal_test_data.pickle','rb') as f1:
		temporal_test_data=pickle.load(f1)

	t_model = prepareTemporalModel(img_channels,img_rows,img_cols,nb_classes)
	f_model = prepareFeaturesModel(nb_classes,requiredLines)

	merged_layer = Merge([t_model, f_model], mode='ave')
	model = Sequential()
	model.add(merged_layer)
	model.load_weights('combined_average_model.h5')
	print 'Compiling model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

	keys=temporal_test_data.keys()
	random.shuffle(keys)
	
	# Starting the training of the final model.
	for chunk in chunks(keys,chunk_size):

		tX_test,tY_test=t_getTrainData(chunk,nb_classes,img_rows,img_cols)
		fX_test,fY_test=f_getTrainData(chunk,nb_classes,requiredLines)
		if (tX_test is not None and fX_test is not None):
				preds = model.predict([tX_test,fX_test])
				print (preds)
				print ('-'*40)
				print (tY_test)

				total_predictions += fX_test.shape[0]
				correct_predictions += totalCorrectPred(preds,tY_test)

				comparisons=[]
				maximum=np.argmax(tY_test,axis=1)
				for i,j in enumerate(maximum):
					comparisons.append(preds[i][j])
				with open('compare.txt','a') as f1:
					f1.write(str(comparisons))
					f1.write('\n\n')
	print "\nThe accuracy was found out to be: ",str(correct_predictions*100/total_predictions)

if __name__ == "__main__":
	CNN()