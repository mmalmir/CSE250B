####################################
### CRF classifier main class    ###
####################################


import numpy as np
import scipy as sp
from transformer import Transformer


class CRFClassifier(Transformer):
    def __init__(self,**kwargs):
        self.xNgramLen   = 2 #length of n-gram, either 1 (xi) or 2 (xi-1 xi)
        self.yNgramLen   = 2 #length of n-gram, either 1 (yi) or 2 (yi-1 yi)
        self.idxToLabel  = dict()
        self.idxToPos    = dict()
        self.trainMethod = "CD"#conrtastive divergence
        self.set_params(**kwargs)


    def set_params(self,**kwargs):
        for k in kwargs:
            if   k=="x_ngram_len":
                self.xNgramLen   = kwargs[k]
            elif k=="y_ngram_len":
                self.yNgramLen   = kwargs[k]
            elif k=="idx_to_label":
                self.idxToLabel  = kwargs[k]
            elif k=="idx_to_pos":
                self.idxToPos    = kwargs[k]
            elif k=="train_method":
                self.trainMethod = kwargs[k]


    #for two given samples x,xhat and given weight vector W,
    #finds the indices of feature functinos Fj that are non zero in either x,xhat,
    #and returns the indices of those non zero Fs and their values
    def nonZeroFeatFuncs(self,x,W):
        bin     = np.bincount(x)
        idx     = np.where(bincount>0)[0]
        return bin[idx],idx


    def fit(self,X,Y,**kwargs):
        self.set_params(**kwargs)
        XCopy = copy.deepcopy(X)
        YCopy = copy.deepcopy(Y)
        if self.trainMethod=="CD":
            self.contrastiveDivergence(XCopy,YCopy)
        elif self.trainMethod=="CollinPerceptron":
            self.CollinPerceptron(XCopy,YCopy)
    
    
    #############################################
    # find yhat using forward backward vectors
    # yhat is the most probable label for x given w
    #############################################
    def mostProbableY(self,x,y,W):

    
    #############################################
    #sample y* by starting from y, randomly changing
    # tags and accepting the change if only the new y
    # has a higher probability
    #############################################
    def sampleY(self,x,y,W):
    
    #############################################
    #train the model using Collin's perceptron
    # wj <- wj + landa x [ Fj(x,y) - Fj(x,y*) ]
    # y* is the evil twin
    #############################################
    def contrastiveDivergence(self,X,Y):
        J = len(self.idxToPos.keys())**self.xNgramLen * (len(self.idxToLabel.keys())**self.yNgramLen)
        #initialize w
        W = 0.0001 * np.random.randn(J)
        n,d = X.shape
        converged = False
        #shuffle input samples
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        sampeleCntr = 0
        landa = 1.
        #repeat until convergence
        while not converged:
            #pick next sample
            x           = X[sampeleCntr,:]
            y           = Y[sampleCntr,:]
            sampelCntr  += 1
            #calculate yhat
            yhat,xhat   = self.sampleY(x,y,W)
            #calculate Fj(x,y) and Fj(x,yhat)
            F,Fidx            = self.nonZeroFeatFuncs(x,W)
            Fh,Fhidx          = self.nonZeroFeatFuncs(xhat,W)
            idxTotal          = np.concatenate([Fidx,diffIdx])
            Fnew,FHnew        = np.zeros(len(idxTotal)),np.zeros(len(idxTotal))
            cnt               = 0
            for i in idxTotal:
                i1,i2 = np.where(Fidx==i)[0],np.where(Fhidx==i)[0]
                if len(i1)==1:
                    Fnew[cnt] = F[i1]
                if len(i2)==1:
                    FHnew[cnt] = Fh[i2]
                cnt += 1
            #update w
            W[Fidx]     = W[Fidx] + landa * (Fnew - FHnew)
            #check for convergence
            Ypredicted = self.predictLabel(X,W)
            pCorrect = (Y!=Ypredicted).sum() / (Y.shape[0]*Y.shape[1])
            if pCorrect>0.99:
                converged = True
        self.W = w

    #############################################
    #train the model using Collin's perceptron
    # wj <- wj + landa x [ Fj(x,y) - Fj(x,yhat) ]
    #############################################
    def CollinPerceptron(self,X,Y):
        J = len(self.idxToPos.keys())**self.xNgramLen * (len(self.idxToLabel.keys())**self.yNgramLen)
        #initialize w
        W = 0.0001 * np.random.randn(J)
        n,d = X.shape
        converged = False
        #shuffle input samples
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        sampeleCntr = 0
        landa = 1.
        #repeat until convergence
        while not converged:
            #pick next sample
            x           = X[sampeleCntr,:]
            y           = Y[sampleCntr,:]
            sampelCntr  += 1
            #calculate yhat
            yhat,xhat   = self.mostProbableY(x,W)
            #calculate Fj(x,y) and Fj(x,yhat)
            F,Fidx            = self.nonZeroFeatFuncs(x,W)
            Fh,Fhidx          = self.nonZeroFeatFuncs(xhat,W)
            idxTotal          = np.concatenate([Fidx,diffIdx])
            Fnew,FHnew        = np.zeros(len(idxTotal)),np.zeros(len(idxTotal))
            cnt               = 0
            for i in idxTotal:
                i1,i2 = np.where(Fidx==i)[0],np.where(Fhidx==i)[0]
                if len(i1)==1:
                    Fnew[cnt] = F[i1]
                if len(i2)==1:
                    FHnew[cnt] = Fh[i2]
                cnt += 1
            #update w
            W[Fidx]     = W[Fidx] + landa * (Fnew - FHnew)
            #check for convergence
            Ypredicted = self.predictLabel(X,W)
            pCorrect = (Y!=Ypredicted).sum() / (Y.shape[0]*Y.shape[1])
            if pCorrect>0.99:
                converged = True
        self.W = w
