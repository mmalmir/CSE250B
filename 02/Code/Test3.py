#usr/bin/python2.7

from __future__ import print_function
import numpy as np
import scipy as sc
import nltk as nltk
from nltk.corpus import brown

def main():
    traininglabel = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/trainingLabels.txt', 'r')
    testdata = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/testSentences.txt', 'r')
    testlabel = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/testLabels.txt', 'r')
    trainingdata = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/trainingSentences.txt', 'r')
    trainingpos = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/trainingSentencesPOS.txt', 'w')
    testpos = open('/Users/Erfan/University/Quarter 2/LearningA/HW/HW2/CSE250B/02/Data/punctuationDataset/testingSentencesPOS.txt', 'w')
    dataline = trainingdata.readlines()
    labelline=traininglabel.readlines()
    testline=testdata.readlines()
   # for i in range(len(dataline)):
   #     print(dataline[i])
   #     print(labelline[i])
    buffersize = 1000
    for i in range(len(testline)):
        token = nltk.word_tokenize(testline[i])
        pos=nltk.pos_tag(token)
        print(pos,file=testpos)
        print('\n',file=testpos)

    trainingpos.close()        
    
if __name__ == "__main__": main()