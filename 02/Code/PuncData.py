import nltk
import re
import nltk
import os
import string
import numpy as np
from topia.termextract import tag
from itertools import chain

class PuncData:
    
    def __init__(self,**kwargs):
        self.dataPath          = "../Data/punctuationDataset/"
        self.trainSen          = self.dataPath+"trainingSentences.txt"
        self.trainLabels       = self.dataPath+"trainingLabels.txt"
        self.testSen           = self.dataPath+"testSentences.txt"
        self.testLabels        = self.dataPath+"testLabels.txt"
        self.set_params(**kwargs)

    
    def set_params(self,**kwargs):
        for k in kwargs:
            if k=="path":
                self.dataPath = kwargs[k]


    def load(self):
        #load sentences
        self.trainSen    = self.loadTextFile(self.trainSen)
        self.trainPos    = self.calculatePosTag(self.trainSen)#convert sentences to pos tags
        self.trainLabels = self.loadTextFile(self.trainLabels)
        n_train          = len(self.trainSen.keys())
        d_train          = np.max([len(l) for l in self.trainSen.values()])

        self.testSen     = self.loadTextFile(self.testSen)
        self.testPos     = self.calculatePosTag(self.testSen)#convert sentences to pos tags
        self.testLabels  = self.loadTextFile(self.testLabels)
        n_test           = len(self.testSen.keys())
        d_test           = np.max([len(l) for l in self.testSen.values()])
        #initialize numeric feature vector
        d                = max(d_train,d_test)
        X_train          = -1 * np.ones([n_train,d])
        Y_train          = -1 * np.ones([n_train,d])
        X_test           = -1 * np.ones([n_test,d])
        Y_test           = -1 * np.ones([n_test,d])
        #gather all pos tags
        self.posTags,self.posToIdx,self.idxToPos       = self.labelToIdx(self.trainPos,self.testPos)
        self.yLabels,self.yLabelToIdx,self.idxToYlabel = self.labelToIdx(self.trainLabels,self.testLabels)
        print self.yLabels,self.yLabelToIdx,self.idxToYlabel




    def labelToIdx(self,l1,l2):
        l1Tags     = [x.values() for x in l1.values()]
        l1Tags     = set([t for t in chain(*l1Tags)])
        l2Tags     = [x.values() for x in l2.values()]
        l2Tags     = set([t for t in chain(*l2Tags)])
        u          = l1Tags.union(l2tags)
        #convert pos tags to numeric indices
        lblToIdx    = dict()
        idxTolbl    = dict()
        i                = 0
        for x in u:
            lblToIdx[x] = i
            idxTolbl[i] = x
            i                += 1
        return (u,lblToIdx,idxTolbl)
    
    
    
    #given a set of sentences, converts each word to pos tag
    def calculatePosTag(self,sentences):
        tagger = tag.Tagger()
        tagger.initialize()
        tags   = dict()
        for k in sentences.keys():
            tags[k] = dict()
            for i in sentences[k].keys():
                token       = sentences[k][i]
                tags[k][i]  = tagger(token)[0][1]
            if len(tags[k].keys())!=len(sentences[k].keys()):
                print tokens,tags
        return tags
#        #fnding unique labels
#        allPos = []
#        for k in tags.keys():
#            for i in tags[k].keys():
#                allPos.append(tags[k][i][0][1])
#        self.allPos = set(allPos)
#        self.allLabels = set([x for x in chain(*labels)])
#        for k in tags.keys():
#            for i in tags[k].keys():
#                X_train[k,i] = tagToInd(tags[k][i][0][1])

            


    #load data from sentences and labels files
    def loadTextFile(self,fname):
        f = open(fname,"rt")
        cnt = 0
        lines = dict()
        for line in f:
            lines[cnt] = dict()
            tokens = re.split(" ",line.rstrip(os.linesep))# re.split(" ",line)
            i = 0
            for token in tokens:
                lines[cnt][i] = token
                i += 1
            cnt += 1
        f.close()
        return lines

