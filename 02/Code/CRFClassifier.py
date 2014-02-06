####################################
### CRF classifier main class    ###
####################################


import numpy as np
from transformer import Transformer


class CRFClassifier(Transformer):
    def __init__(self,**kwargs):
        self.xNgramLen   = 2 #length of n-gram, either 1 (xi) or 2 (xi-1 xi)
        self.yNgramLen   = 2 #length of n-gram, either 1 (yi) or 2 (yi-1 yi)
        self.idxToLabel  = dict()
        self.idxToPos    = dict()
        self.set_params(**kwargs)


    def set_params(self,**kwargs):
        for k in kwargs:
            if k=="x_ngram_len":
                self.xNgramLen = kwargs[k]
            elif k=="y_ngram_len":
                self.yNgramLen = kwargs[k]
            elif k=="idx_to_label":
                self.idxToLabel = kwargs[k]
            elif k=="idx_to_pos":
                self.idxToPos = kwargs[k]


    #############################################
    #train the model using Collin's perceptron
    # wj <- wj + landa x [ Fj(x,y) - Fj(x,yhat) ]
    #############################################
    def CollinPerceptron(self,X,Y,**kwargs):
        self.set_params(**kwargs)
        J = len(self.idxToPos.keys())**self.xNgramLen * (len(self.idxToLabel.keys())**self.yNgramLen)
        #initialize w
        W = 0.0001 * np.random.randn(J)
        #repeat until convergence
        n,d = X.shape
        converged = False
        idx = np.arange(n)
        np.random.shuffle(idx)
        print idx
        sampeleCntr = 0
        landa = 1.
        while not converged:
            #pick next sample
            x = X[sampeleCntr,:]
            y = Y[sampleCntr,:]
            sampelCntr += 1
            #calculate yhat
            yhat,xhat = self.mostProbableY(x,W)
            #calculate Fj(x,y) and Fj(x,yhat)
            Fidx,Fhat,F = self.featFunc(xhat,W),self.featFunc(x,W)
            #update w
            W[Fidx] = W[Fidx] + landa * (F - Fhat)
            #check for convergence
            
