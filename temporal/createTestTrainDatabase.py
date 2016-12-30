import os
from optical_flow_prep import writeOpticalFlow
import pickle
import sys

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

# Produces optical flow images and creates a dictionary of the files present
# with their respective classes. Pass --test argument for preparing testing data
# pickle file. 
def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        pickleFile = "../dataset/temporal_test_data.pickle"
        fileList = "../dataset/ucfTrainTestlist/testlist01.txt"
        print "Saving data at: ",pickleFile
    else:
        pickleFile = "../dataset/temporal_train_data.pickle"
        fileList = "../dataset/ucfTrainTestlist/trainlist01.txt"
        print "Saving data at: ",pickleFile
    training_data = {}
    types = CreateDict("../dataset/ucfTrainTestlist/classInd.txt")
    rootDir = '../dataset/ucf101/'
    f = open(fileList,'r')       
    for line in f:
        filePath = line.split()[0]
        folder,file = filePath.split('/')
        count = writeOpticalFlow(rootDir+folder,file,150,150,1)
        print count
        blockno = int(count/50)
        key = file + '@' + str(blockno)
        training_data[key] = types[folder]
    print training_data
    with open(pickleFile,'wb') as f:
        pickle.dump(training_data,f)


if __name__ == '__main__':
    main()
