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

def f_getTrainData(chunk,nb_classes):
	X_train,Y_train = efp.stackExtractedFeatures(chunk,'train')
	if (X_train is not None and Y_train is not None):
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def t_getTrainData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOF(chunk,img_rows,img_cols,'train')
	if (X_train is not None and Y_train is not None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def prepareFeaturesModel(nb_classes):
	print "Preparing architecture of feature model..."
	f_model = Sequential()
	f_model.add(Dense(512, input_shape=(167,10)))
	f_model.add(Flatten())
	f_model.add(Dense(512, W_regularizer=l2(0.1)))

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

	t_model.add(Dense(512))

	return t_model


def CNN():
	input_frames=10
	batch_size=10
	nb_classes = 3
	nb_epoch = 1
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=8

	print 'Loading dictionary...'
	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	t_model = prepareTemporalModel(img_channels,img_rows,img_cols,nb_classes)
	f_model = prepareFeaturesModel(nb_classes)

	merged_layer = Merge([t_model, f_model], mode='ave')
	model = Sequential()
	model.add(merged_layer)
	model.add(Dense(nb_classes, W_regularizer=l2(0.01)))
	model.add(Activation('softmax'))

	print 'Compiling model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	model.compile(loss='hinge',optimizer=sgd, metrics=['accuracy'])

	keys=temporal_train_data.keys()
	random.shuffle(keys)
	
	# Starting the training of the final model.
	instance_count=0
	for e in range(nb_epoch):
		print('-'*40)
		print('Epoch', e)
		print('-'*40)

		flag=0
		keys=temporal_train_data.keys()
		random.shuffle(keys)

		for chunk in chunks(keys,chunk_size):
			if flag<1:
				print("Preparing testing data...")
				tX_test,tY_test=t_getTrainData(chunk,nb_classes,img_rows,img_cols)
				fX_test,fY_test=f_getTrainData(chunk,nb_classes)
				flag += 1
				continue
			print "Instance count :", instance_count
			instance_count+=chunk_size
			tX_batch,tY_batch=t_getTrainData(chunk,nb_classes,img_rows,img_cols)
			fX_batch,fY_batch=f_getTrainData(chunk,nb_classes)

			if (tX_batch is not None and fX_batch is not None):
				loss = model.fit([tX_batch, fX_batch], tY_batch, verbose=1, batch_size=batch_size, nb_epoch=1)
				if instance_count%160==0:
					loss = model.evaluate([tX_test,fX_test],tY_test,batch_size=batch_size,verbose=1)
					preds = model.predict([tX_test,fX_test])
					print (preds)
					print ('-'*40)
					print (tY_test)
					comparisons=[]
					maximum=np.argmax(tY_test,axis=1)
					for i,j in enumerate(maximum):
						comparisons.append(preds[i][j])
					with open('compare.txt','a') as f1:
						f1.write(str(comparisons))
						f1.write('\n\n')
					with open('loss.txt','a') as f1:
						f1.write(str(loss))
						f1.write('\n')
					model.save_weights('combined_merge_model.h5',overwrite=True)
					return
if __name__ == "__main__":
	CNN()
