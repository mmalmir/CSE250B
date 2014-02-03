import nltk
import re
from itertools import chain

class PuncData(Transformer):
    
    def __init__(self,**kwargs):
        self.dataPath          = "../Data/punctuationDataset/"
        self.trainSen          = "trainingSentences.txt"
        self.trainLabels       = "trainingLabels.txt"
        self.trainSenPos       = "trainingSentencesPOS.txt"
        self.testSen           = "testSentences.txt"
        self.testLabels        = "testLabels.txt"
        self.testSenPos        = "testingSentencesPOS.txt"
        self.set_params(**kwargs)

    
    def set_params(self,**kwargs):
        for k in kwargs:
            if k=="path":
                self.dataPath = kwargs[k]


    def load(self):
    #load sentences
        trainSen    = self.loadTextFile(self.trainSen)
        trainLabels = self.loadTextFile(self.trainLabels)
        trainPos    = self.loadTextFile(self.trainingSentencesPOS)
        n_train     = len(trainSen.keys())
        d_train     = np.max([len(l) for l in trainSen.values()])

        testSen     = self.loadTextFile(self.testSen)
        testLabels  = self.loadTextFile(self.testLabels)
        testPos     = self.loadTextFile(self.testingSentencesPOS)
        n_test      = len(testSen.keys())
        d_test      = np.max([len(l) for l in testSen.values()])
        #create numeric feature vector
        d           = max(d_train,d_test)
        X_train     = -1 * np.ones([n_train,d])
        Y_train     = -1 * np.ones([n_train,d])
        X_test      = -1 * np.ones([n_test,d])
        Y_test      = -1 * np.ones([n_test,d])
        #fnding unique labels
        allPos      = [x for x in chain(*trainPos.values())]
        pos         = set()


    #load data from sentences and labels files
    def loadTextFile(self,fname):
        f = fopen(fname,"rt")
        cnt = 1
        lines = dict()
        for line in f:
            lines[cnt] = re.split(" ",line)
            cnt += 1
        f.close()
        return lines

