import numpy as np
import h5py
import gc
import extracted_features_prep as efp
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]


def compileFeaturesModel(nb_classes, requiredLines):
	f_model = Sequential()
	f_model.add(Dense(512, input_shape=(167,requiredLines)))
	f_model.add(Flatten())
	f_model.add(Dense(nb_classes, W_regularizer=l2(0.1)))
	f_model.add(Activation('linear'))
	f_model.add(Activation('softmax'))

	print 'Compiling f_model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	f_model.compile(loss='hinge',optimizer=sgd, metrics=['accuracy'])
	return f_model

def f_getTrainData(chunk,nb_classes,requiredLines):
	X_train,Y_train = efp.stackExtractedFeatures(chunk,'train',requiredLines)
	if (X_train is not None and Y_train is not None):
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)

def CNN():
	batch_size= 10
	nb_classes = 3
	nb_epoch = 5
	chunk_size=8
	requiredLines = 1000
	print 'Loading dictionary...'

	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	f_model = compileFeaturesModel(nb_classes, requiredLines)

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
				X_test,Y_test=f_getTrainData(chunk,nb_classes,requiredLines)
				flag += 1
				continue
			print "Instance count :", instance_count
			instance_count+=chunk_size
			X_batch,Y_batch=f_getTrainData(chunk,nb_classes,requiredLines)
			if (X_batch is not None and Y_batch is not None):
				loss = f_model.fit(X_batch, Y_batch, verbose=1, batch_size=batch_size, nb_epoch=1)	
				if instance_count%160==0:
					loss = f_model.evaluate(X_test,Y_test,batch_size=batch_size,verbose=1)
					preds = f_model.predict(X_test)
					print (preds)
					print ('-'*40)
					print (Y_test)
					comparisons=[]
					maximum=np.argmax(Y_test,axis=1)
					for i,j in enumerate(maximum):
						comparisons.append(preds[i][j])
					with open('compare.txt','a') as f1:
						f1.write(str(comparisons))
						f1.write('\n\n')
					with open('loss.txt','a') as f1:
						f1.write(str(loss))
						f1.write('\n')
					f_model.save_weights('features_stream_model.h5',overwrite=True)
			else : 
				print "features_stream_cnn.py: X or Y_batch is None"

if __name__ == "__main__":
	CNN()
