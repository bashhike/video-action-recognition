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

# Read some predefined lines from the file. If the lines in the file fall 
# short, continue reading from the begining.
def getlines(f,requiredLines):
	f.seek(0)
	lineCount = 0
	flag = True
	for line in f:
		if line[0] == '#': continue
		if lineCount >= requiredLines : break
		if flag:
			flag = False
			inp = np.array(line.split()[4:],dtype='float16')
			lineCount += 1
			continue
		temp = np.array(line.split()[4:],dtype='float16')
		inp = np.dstack((inp,temp))
		lineCount += 1
	while lineCount < requiredLines :
		line = f.readline()
		if(len(line) is not 0):
			if line[0] == '#': continue
			temp = np.array(line.split()[4:],dtype='float16')
			inp = np.dstack((inp,temp))
			lineCount += 1
		else:
			f.seek(0)
	return inp

# Read the *HOGHOF.txt files and stack them into numpy arrays
def stackExtractedFeatures(chunk,jobtype,requiredLines):
	firstTime=1
	types = CreateDict("../dataset/ucfTrainTestlist/classInd.txt")
	try:
		firstTimeOuter=1
		for item in chunk:
			filename,itemNo=item.split('@')
			folder = filename.split('_')[1]
			filepath = '../dataset/ucf101/'+folder+'/'+filename.split('.')[0]+'_HOGHOF.txt'
			f = open(filepath,'r')
			inp = getlines(f,requiredLines)
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
		return (inputVec,labels)
	except : 
		print "Extracted_features.py: Some error encountered, returning (None,None)"
		return (None,None)