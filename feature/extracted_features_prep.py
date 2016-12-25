import cv2
import numpy as np
import pickle
from PIL import Image
import os
import gc

# Creates a dictionary of the Classes of Videos present in the database
# and assigns them value starting from 0 alphabetically
def CreateDict(filename):
    f = open(filename,'r')
    a = f.readlines()
    d = {}
    count = 1
    for i in a:
        temp = i.split()
        i = temp[-1]
        d[i] = int(temp[0])
        count += 1
    return d

def stackExtractedFeatures(chunk,jobtype):
	firstTime=1
	types = CreateDict("../dataset/ucfTrainTestlist/classInd.txt")
	try:
		firstTimeOuter=1
		for item in chunk:
			filename,itemNo=item.split('@')
			folder = filename.split('_')[1]
			filepath = '../dataset/ucf101/'+folder+'/'+filename.split('.')[0]+'_HOGHOF.txt'
			f = open(filepath,'r')
			data = f.readlines() 
			inp = np.array(data[-1].split()[4:],dtype='float16')
			for i in range(2,11):
				temp = np.array(data[-1*i].split()[4:],dtype='float16')
				inp = np.dstack((inp,temp))
			if not firstTime:
				inputVec = np.concatenate((inputVec,inp))
				labels=np.append(labels,types[folder]-1)
			else:
				inputVec = inp
				labels=np.array(types[folder]-1)
				firstTime = 0

		inputVec=inputVec.astype('float16',copy=False)
		labels=labels.astype('int',copy=False)
		gc.collect()

		#print "extracted_features_prep.py : ",inputVec.shape, labels.shape
		return (inputVec,labels)
	except:
		print "extracted_features_prep.py : Some exception occured, returning (None,None)"
		return (None,None)
