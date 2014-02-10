####################################
### CRF classifier main class    ###
####################################

import copy
import pprint
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
        xCopy   = x[np.where(x>0)[0]]
        bin     = np.bincount(xCopy)
        idx     = np.where(bin>0)[0]
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
    def mostProbableY(self,x,W):
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        A             = x[0]/(numYs*numPos)
        B             = (x[0]-A*numYs*numPos)/numYs
        #initialize U(y,1) as g_0(START,y)
        idx           = A*numYs*numPos+B*numYs+startY*numLabels+np.arange(numLabels)
        U             = W[idx].reshape([-1,1])
        bestY         = startY*np.ones(numLabels).reshape([-1,1]).astype(np.int)
        temp          = np.arange(numLabels).reshape([1,-1])
        temp2         = np.arange(numLabels).reshape([-1,1])
        bestYs        = dict()
        bestYs[0]     = copy.deepcopy(bestY.reshape(-1))
        Us            = dict()
        Us[0]         = copy.deepcopy(U)
        i             = 0
        finished      = False
#        print x
        while not finished:
            i         += 1
            A         = x[i]/(numPos*numYs)
            B         = (x[i]-A*numPos*numYs)/numYs
            idx       = A*numYs*numPos+B*numYs
            # idx is a 2d matrix: idx_ij = g_i(best_y_i-1,y_i)
            if x[i+1]==0:
                finished  = True
                idx       = idx + np.tile(temp2*numLabels,[1,numLabels]) # + 0 (STOP)
            else:
                idx       = idx + np.tile(temp2*numLabels,[1,numLabels]) + np.tile(temp,[numLabels,1])
            temp3     = np.argmax(W[idx]+np.tile(U,[1,numLabels]),axis=0)
#            print temp2
            bestY     = copy.deepcopy(temp3).reshape([-1,1])
#            print temp3
#            print idx
#            print x
#            print "\n"
#            print W[idx[np.arange(numLabels),temp2]]
            # convert function indices to label indices
            U         = W[idx[np.arange(numLabels),temp3]].reshape([-1,1])+U[temp3].reshape([-1,1])
#            U         = U[temp2].reshape([-1,1])
            bestYs[i] = copy.deepcopy(bestY.reshape(-1))
            Us[i]     = copy.deepcopy(U.reshape(-1))
        lenSentence  = i
#        print bestYs
        #reconstruct the best solution
        Y       =  0*np.ones(x.shape[0])
        idx     =  -1
        yidx    =  lenSentence-1
        keys    =  np.sort(-np.asarray(Us.keys()))
        U       =  Us[-keys[0]]
        idx     =  np.argmax(U)
        Y[yidx] =  idx
        yidx    -= 1
        for i in keys[1:]:
            bestY = bestYs[-i]
            idx = bestY[idx]
            Y[yidx] =  idx
            yidx    -= 1
            if yidx<0:
                break
        #finding xhat
        finished  = False
        xhat = copy.deepcopy(x)
        A         = x[0]/(numPos*numYs)
        B         = (x[0]-A*numPos*numYs)/numYs
        C         = startY
        D         = Y[0]
        xhat[0]   = A*numPos*numYs+B*numYs+C*numLabels+D
        i         = 0
        while not finished:
            i         += 1
            A         = x[i]/(numPos*numYs)
            B         = (x[i]-A*numPos*numYs)/numYs
            C         = Y[i-1]
            if x[i+1]==0:
                finished=True
                D         = 0
            else:
                D         = Y[i]
            xhat[i]   = A*numPos*numYs+B*numYs+C*numLabels+D
        return (np.asarray(Y),xhat)


    #############################################
    # update x using the new y
    #############################################
    def updateX(self,x,y):
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        finished  = False
        xhat = copy.deepcopy(x)
        A         = x[0]/(numPos*numYs)
        B         = (x[0]-A*numPos*numYs)/numYs
        C         = startY
        D         = y[0]
        xhat[0]   = A*numPos*numYs+B*numYs+C*numLabels+D
        i         = 0
        while not finished:
            i         += 1
            A         = x[i]/(numPos*numYs)
            B         = (x[i]-A*numPos*numYs)/numYs
            C         = y[i-1]
            if x[i+1]==0:
                finished=True
                D         = 0
            else:
                D         = y[i]
            xhat[i]   = A*numPos*numYs+B*numYs+C*numLabels+D
        return xhat
        
    #############################################
    #sample y* by starting from y, randomly changing
    # tags and accepting the change if  the new y
    # has a higher probability
    #############################################
    def sampleY(self,x,y,W):
        numLabels = len(self.idxToLabel.keys())
        l         = np.where(x>0)[0].shape[0]
        n         = l/3
        finished  = False
        pCurrent = W[x[np.where(x>0)[0]]].sum()
        while not finished:
            #select n rangom positions in y
            newY = copy.deepcopy(y)
            idx = np.arange(l)
            np.random.shuffle(idx)
            idx = idx[:n]
#            print idx
            #flip n tags
            labelIdx  = np.random.randint(0,numLabels,n)
            newY[idx] = labelIdx
            newX      = self.updateX(x,newY)
            #accept if y_new has higher probability
            pNew = W[newX[np.where(newX>0)[0]]].sum()
#            print x
#            print newX
#            print pNew,pCurrent
#            if pNew>pCurrent:
            finished = True
        return newX.astype(np.int),newY.astype(np.int)

    #############################################
    #predicts the most probable label for X
    #############################################
    def predictLabel(self,X,W):
        n,d = X.shape
        Y   = np.zeros(X.shape)
        for i in range(n):
            Y[i,:],_ = self.mostProbableY(X[i,:],W)
        return Y

    #############################################
    #train the model using Collin's perceptron
    # wj <- wj + landa x [ Fj(x,y) - Fj(x,yhat) ]
    #############################################
    def CollinPerceptron(self,X,Y):
        numPos    = len(self.idxToPos.keys())+2
        numLabels = len(self.idxToLabel.keys())+2
        J = numPos**self.xNgramLen * numLabels**self.yNgramLen
        #initialize w
        W   = 0.00001*np.random.randn(J)
        n,d = X.shape
        converged = False
        #shuffle input samples
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        sampleCntr = 0
        landa = 1.
        #repeat until convergence
        while not converged:
            if sampleCntr%n==n-1:
                Ypredicted = self.predictLabel(X,W)
                pCorrect = (Y==Ypredicted).sum()/float(Y.shape[1]*Y.shape[0])
#                print np.where(Y!=Ypredicted)
#                pCorrect = (Y[:,1:]==Ypredicted[:,1:]).sum()/float((Y.shape[1]-1)*Y.shape[0])
#                idxxx = np.any(Y!=Ypredicted,axis=1)
#                print Y[idxxx,:]-Ypredicted[idxxx,:]
#                pCorrect = 1. - np.any(Y!=Ypredicted,axis=1).sum()/float(n)
#                for i in range(n):
#                    print Y[i,:]
#                    print Ypredicted[i,:]
#                    print "\n"
                print pCorrect
                if pCorrect>0.99:
                    converged = True
            #pick next sample
            x                 =  X[sampleCntr,:]
            y                 =  Y[sampleCntr,:]
            sampleCntr        += 1
            sampleCntr        = sampleCntr % n
            #calculate yhat
            yhat,xhat         =  self.mostProbableY(x,W)
#            print x
#            print y
#            print xhat
#            print yhat
#            print W[x[np.where(x>0)[0]]]
#            print "\n"
            #calculate Fj(x,y) and Fj(x,yhat)
            F,Fidx            =  self.nonZeroFeatFuncs(x,W)
            Fh,Fhidx          =  self.nonZeroFeatFuncs(xhat,W)
            idxTotal          =  np.unique(np.concatenate([Fidx,Fhidx]))
            Fnew,FHnew        =  np.zeros(len(idxTotal)),np.zeros(len(idxTotal))
            cnt               =  0
            for i in idxTotal:
                i1,i2 = np.where(Fidx==i)[0],np.where(Fhidx==i)[0]
                if len(i1)==1:
                    Fnew[cnt] = F[i1]
                elif len(i1)>1:
                    print i1
                if len(i2)==1:
                    FHnew[cnt] = Fh[i2]
                elif len(i2)>1:
                    print i2
                cnt += 1
            #update w
#            print x
#            print idxTotal
#            print landa * (Fnew - FHnew)
##            print Fnew
##            print FHnew
#            print W[idxTotal]
#            print "\n"
            W[idxTotal]     = W[idxTotal] + landa * (Fnew - FHnew)
#            print (W[np.where(W>0)[0]])
#            print landa * (Fnew - FHnew)
        self.W = W

    #############################################
    #train the model using Collin's perceptron
    # wj <- wj + landa x [ Fj(x,y) - Fj(x,y*) ]
    # y* is the evil twin
    #############################################
    def contrastiveDivergence(self,X,Y):
        numPos    = len(self.idxToPos.keys())+2
        numLabels = len(self.idxToLabel.keys())+2
        J = numPos**self.xNgramLen * numLabels**self.yNgramLen
        #initialize w
        W   = 0.00001*np.random.randn(J)
        W[0] = 10
        n,d = X.shape
        converged = False
        #shuffle input samples
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        sampleCntr = 0
        landa = 1.
        #repeat until convergence
        while not converged:
#            print "one"
            #pick next sample
            x           = X[sampleCntr,:]
            y           = Y[sampleCntr,:]
            sampleCntr  += 1
            sampleCntr  %= n
            #calculate yhat
            yhat,xhat   = self.sampleY(x,y,W)
            #calculate Fj(x,y) and Fj(x,yhat)
            F,Fidx            =  self.nonZeroFeatFuncs(x,W)
            Fh,Fhidx          =  self.nonZeroFeatFuncs(xhat,W)
            idxTotal          =  np.unique(np.concatenate([Fidx,Fhidx]))
            Fnew,FHnew        =  np.zeros(len(idxTotal)),np.zeros(len(idxTotal))
            cnt               =  0
            for i in idxTotal:
                i1,i2 = np.where(Fidx==i)[0],np.where(Fhidx==i)[0]
                if len(i1)==1:
                    Fnew[cnt] = F[i1]
                elif len(i1)>1:
                    print i1
                if len(i2)==1:
                    FHnew[cnt] = Fh[i2]
                elif len(i2)>1:
                    print i2
                cnt += 1
            #update w
            W[idxTotal]     = W[idxTotal] + landa * (Fnew - FHnew)
            #check for convergence
            if sampleCntr%n==n-1:
                Ypredicted = self.predictLabel(X,W)
                pCorrect   = (Y==Ypredicted).sum()/float(Y.shape[1]*Y.shape[0])
#                print np.where(Y!=Ypredicted)
#                pCorrect = (Y[:,1:]==Ypredicted[:,1:]).sum()/float((Y.shape[1]-1)*Y.shape[0])
#                idxxx = np.any(Y!=Ypredicted,axis=1)
#                print Y[idxxx,:]-Ypredicted[idxxx,:]
#                pCorrect = 1. - np.any(Y!=Ypredicted,axis=1).sum()/float(n)
#                for i in range(n):
#                    print Y[i,:]
#                    print Ypredicted[i,:]
#                    print "\n"
                print pCorrect
                if pCorrect>0.99:
                    converged = True
        self.W = W
