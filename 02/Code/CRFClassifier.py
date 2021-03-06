####################################
### CRF classifier main class    ###
####################################

import copy
import pprint
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from transformer import Transformer


class CRFClassifier(Transformer):
    def __init__(self,**kwargs):
        self.xNgramLen   = 2 #length of n-gram, either 1 (xi) or 2 (xi-1 xi)
        self.yNgramLen   = 2 #length of n-gram, either 1 (yi) or 2 (yi-1 yi)
        self.turn        = 0
        self.idxToLabel  = dict()
        self.idxToPos    = dict()
        self.trainMethod = "CD"#conrtastive divergence
        self.W           = None
        self.turn        = 0
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
            elif k=="sampling":
                self.samplingMethod = kwargs[k]
            elif k=="turn":
                self.turn = kwargs[k]


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
        return (np.asarray(Y).astype(np.int),xhat.astype(np.int))


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
#        print x[0],xhat[0],A,B,C,D
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
    #sample y* by from posterior P(y_idx | y_{-idx})
    #############################################
    def posterior(self,x,y,W,idx):
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        l             = np.where(x>0)[0].shape[0]
        yNew          = copy.deepcopy(y)
        for i in idx:
            A = x[i]/(numPos*numYs)
            B = (x[i]-A*numPos*numYs)/numYs
            if i==0:
                yPre = startY
            else:
                yPre = y[i-1]
            if i==l-2:
                yNext = 0
            else:
                yNext = y[i+1]
            Py  = np.zeros(numLabels)
            for yy in range(numLabels):
                  g_i       = W[A*numPos*numYs+B*numYs+yPre*numLabels+yy]
                  g_iP1     = W[A*numPos*numYs+B*numYs+yy*numLabels+yNext]
                  Py[yy]    = np.exp(g_i)*np.exp(g_iP1)
            Py = Py / Py.sum()
            #select a sample
            newSample = np.random.multinomial(1,Py)
            yNew[i] = np.where(newSample)[0][0]
        return yNew

    
    #############################################
    #sample y* by from posterior P(y_idx | y_{-idx})
    #############################################
    def guidedSample(self,x,y,W):
        idx           = np.where((y!=2) * (y!=4) * (y!=0))[0]
        if idx.shape[0]>0:
            idx       = [idx[np.random.randint(0,idx.shape[0])]]
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        l             = np.where(x>0)[0].shape[0]
        yNew          = copy.deepcopy(y)
        for i in idx:
            A = x[i]/(numPos*numYs)
            B = (x[i]-A*numPos*numYs)/numYs
            if i==0:
                yPre = startY
            else:
                yPre = y[i-1]
            if i==l-2:
                yNext = 0
            else:
                yNext = y[i+1]
            Py  = np.zeros(numLabels)
            for yy in range(numLabels):
                if yy not in [2,4]:
                    Py[yy] = 0
                else:
                    g_i       = W[A*numPos*numYs+B*numYs+yPre*numLabels+yy]
                    g_iP1     = W[A*numPos*numYs+B*numYs+yy*numLabels+yNext]
                    Py[yy]   = np.exp(g_i)*np.exp(g_iP1)
            Py = Py / Py.sum()
            #select a sample
            newSample = np.random.multinomial(1,Py)
            yNew[i] = np.where(newSample)[0][0]
        return yNew
    
    
    
    #############################################
    #sample y* by starting from y, randomly changing
    # tags and accepting the change if  the new y
    # has a higher probability
    #############################################
    def sampleY(self,x,y,W,method):
        numLabels = len(self.idxToLabel.keys())+2
        l         = np.where(x>0)[0].shape[0]
        n         = 1
        finished  = False
        pCurrent = W[x[np.where(x>0)[0]]].sum()
        cntr = 0
        while not finished:
            #select n rangom positions in y
            newY = copy.deepcopy(y)
            #flip n tags
            if method=="random":
                if self.turn==-1:
                    idx  = np.arange(l)
                    np.random.shuffle(idx)
                    idx = idx[:n]
                    labelIdx  = np.random.randint(0,numLabels,idx.shape[0])
                    newY[idx] = labelIdx
                else:
                    idx  = np.where(y==self.turn)[0]#COLON
                    labelIdx  = np.random.randint(0,numLabels,idx.shape[0])
                    if idx.shape[0]>0:
                        newY[idx] = labelIdx
                newX = self.updateX(x,newY)
                finished = True
            elif method=="guided":
                idx  = np.arange(l)
                np.random.shuffle(idx)
                idx = idx[:n]
                idxnonSpace = np.where((y!=2) * (y!=4) * (y!=0))[0]
                if idxnonSpace.shape[0]==0:
                    newY      = self.posterior(x,y,W,idx)
                else:
                    newY      = self.guidedSample(x,y,W)
                newX = self.updateX(x,newY)
                finished = True
            elif method=="posterior":
                idx  = np.arange(l)
                np.random.shuffle(idx)
                idx = idx[:n]
                newY = self.posterior(x,y,W,idx)
                newX = self.updateX(x,newY)
                pNew = W[newX[np.where(newX>0)[0]]].sum()
                if pNew>pCurrent or cntr>5:
                    finished = True
            #accept if y_new has higher probability
            cntr += 1
        return newY.astype(np.int),newX.astype(np.int)

    def transform(self,X,**kwargs):
        self.set_params(**kwargs)
        return self.predictLabel(X,self.W)
    
    
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
        if self.W == None:
            W   = 0.00001*np.random.randn(J)
        else:
            W   = self.W
        n,d = X.shape
        converged = False
        #shuffle input samples
        validation = 1./3.
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        vX,vY=X[:int(validation*n),:],Y[:int(validation*n),:]
        X,Y=X[int(validation*n):,:],Y[int(validation*n):,:]
        n,d = X.shape
        sampleCntr = 0
        landa = 1.
        #repeat until convergence
        lastValidationError = 0.
        cntEq = 0
        numEpochs = 0
        trainCorrect = []
        validCorrect = []
        #untrained prediction accuracy
        idxNonZero = np.where(X!=0)
        Ypredicted = self.predictLabel(X,W)
        pCorrect1 = (Y[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Y[idxNonZero].shape[0])
        idxNonZero = np.where(vX!=0)
        Ypredicted = self.predictLabel(vX,W)
        pCorrect = (vY[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(vY[idxNonZero].shape[0])
        trainCorrect.append(pCorrect1)
        validCorrect.append(pCorrect)
        while not converged:
            #pick next sample
            x                 =  X[sampleCntr,:]
            y                 =  Y[sampleCntr,:]
            sampleCntr        += 1
            sampleCntr        = sampleCntr % n
            #calculate yhat
            yhat,xhat         =  self.mostProbableY(x,W)
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
            Wtemp = copy.deepcopy(W)
            W[idxTotal]     = W[idxTotal] + landa * (Fnew - FHnew)
            Xs = np.arange(numPos**2).astype(np.int)*numLabels**2+16
            W[Xs] = 0
            mn,mx = -0.1,0.1
#            for i in idxTotal:
#                if W[i]>mx:
#                    W[i] = mx
#                elif W[i] < mn:
#                    W[i] = mn
            if sampleCntr%n==0:
                numEpochs += 1
                Ypredicted = self.predictLabel(X,W)
                idxNonZero = np.where(X!=0)
                pCorrect1 = (Y[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Y[idxNonZero].shape[0])
                print "training error:",pCorrect1
                

                idxNonZero = np.where(vX!=0)
                Ypredicted = self.predictLabel(vX,W)
                pCorrect = (vY[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(vY[idxNonZero].shape[0])
                print "validation error:",pCorrect
                if pCorrect==lastValidationError:
                    cntEq += 1
                else:
                    cntEq = 0
                if  pCorrect<lastValidationError or numEpochs>5 or cntEq>5:
                    converged = True
                    W = Wtemp
                trainCorrect.append(pCorrect1)
                validCorrect.append(pCorrect)
                lastValidationError = pCorrect
    
    
        ax = plt.figure(1).gca()
        p1, = ax.plot(trainCorrect,"b-x",label="train accuracy")
        p2, = ax.plot(validCorrect,"r--o",label="validation accuracy")
        p3, = ax.plot([len(trainCorrect)-2,len(trainCorrect)-2],[0.,1.],"g-.d",label="early stopping")
        handles, labels = ax.get_legend_handles_labels()
        plt.xlabel("Epoch")
        plt.ylabel("Punctuation Tag Prediction Accuracy")
        ax.legend(handles[::-1], labels[::-1],loc=3)
        plt.savefig("Perceptron.png",dpi=160,bbox_inches="tight",format="png")

    
        #sentence level recognition rates
        Ypredicted = self.predictLabel(vX,W)
        idxNonZero = np.where(vX!=0)
        correct    = np.any(vY!=Ypredicted,axis=1)
        print "sentence level recognition for validation set:",1.-correct.sum()/float(correct.shape[0])
    
        self.W = W
        self.printW(W,20)

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
        if self.W==None:
            W   = 0.0001*np.random.randn(J)
        else:
            W   = self.W
        n,d = X.shape
        converged = False
        #shuffle input samples
        validation = 1./3.
        idx = np.arange(n)
        np.random.shuffle(idx)
        X,Y = X[idx,:],Y[idx,:]
        vX,vY=X[:validation*n,:],Y[:validation*n,:]
        X,Y=X[validation*n:,:],Y[validation*n:,:]
        n,d = X.shape
        sampleCntr = 0
        landa = 1.
        #repeat until convergence
        lastValidationError = 0.
        cntEq = 0
        numEpochs = 0
        trainCorrect = []
        validCorrect = []
        #untrained prediction accuracy
        idxNonZero = np.where(X!=0)
        Ypredicted = self.predictLabel(X,W)
        pCorrect1 = (Y[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Y[idxNonZero].shape[0])
        idxNonZero = np.where(vX!=0)
        Ypredicted = self.predictLabel(vX,W)
        pCorrect = (vY[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(vY[idxNonZero].shape[0])
        trainCorrect.append(pCorrect1)
        validCorrect.append(pCorrect)
        while not converged:
#            print "one"
            #pick next sample
            x           = X[sampleCntr,:]
            y           = Y[sampleCntr,:]
            sampleCntr  += 1
            sampleCntr  %= n
            #calculate yhat
            yhat,xhat   = self.sampleY(x,y,W,self.samplingMethod)
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
            Wtemp = copy.deepcopy(W)
            W[idxTotal]     = W[idxTotal] + landa * (Fnew - FHnew)
#            Xs = np.arange(numPos**2).astype(np.int)*numLabels**2+16
#            W[Xs] = 0
            mn,mx = -0.1,0.1
            for i in idxTotal:
                if W[i]>mx:
                    W[i] = mx
                elif W[i] < mn:
                    W[i] = mn
            #check for convergence
            if sampleCntr%n==0:
                numEpochs += 1
                idxNonZero = np.where(X!=0)
                Ypredicted = self.predictLabel(X,W)
                pCorrect1 = (Y[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Y[idxNonZero].shape[0])
                print "training error:",pCorrect1
                
                idxNonZero = np.where(vX!=0)
                Ypredicted = self.predictLabel(vX,W)
                pCorrect = (vY[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(vY[idxNonZero].shape[0])
                print "validation error:",pCorrect
                if pCorrect==lastValidationError:
                    cntEq += 1
                else:
                    cntEq = 0
                if  pCorrect<lastValidationError or numEpochs>5 or cntEq>5:
                    converged = True
                    W = Wtemp
                trainCorrect.append(pCorrect1)
                validCorrect.append(pCorrect)
                lastValidationError = pCorrect
        self.W = copy.deepcopy(W)
        self.printW(W,20)

        ax = plt.figure(1).gca()
        p1, = ax.plot(trainCorrect,"b-x",label="train accuracy")
        p2, = ax.plot(validCorrect,"r--o",label="validation accuracy")
        p3, = ax.plot([len(trainCorrect)-2,len(trainCorrect)-2],[0.,1.],"g-.d",label="early stopping")
        handles, labels = ax.get_legend_handles_labels()
        plt.xlabel("Epoch")
        plt.ylabel("Punctuation Tag Prediction Accuracy")
        ax.legend(handles[::-1], labels[::-1],loc=3)
        plt.savefig("CD-"+self.samplingMethod+".png",dpi=160,bbox_inches="tight",format="png")

        #sentence level recognition rates
        Ypredicted = self.predictLabel(vX,W)
        idxNonZero = np.where(vX!=0)
        correct    = np.any(vY!=Ypredicted,axis=1)
        print "sentence level recognition for validation set:",1. - correct.sum()/float(correct.shape[0])
    
    
    #print the n largest elements of W
    def printW(self,W,n):
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        indices       = np.argsort(-W)
        newPosDic     = copy.deepcopy(self.idxToPos)
        newPosDic[0]  = "STOP"
        newPosDic[numPos-1]  = "START"
        newLblDic     = copy.deepcopy(self.idxToLabel)
        newLblDic[0]  = "STOP"
        newLblDic[numLabels-1]  = "START"
        for i in range(n):
            idx = indices[i]
            A   = idx/(numPos*numYs)
            B   = (idx-A*numPos*numYs)/numYs
            C   = (idx-A*numPos*numYs-B*numYs)/numLabels
            D   = idx % numLabels
            print W[idx],newPosDic[A],newPosDic[B],newLblDic[C],newLblDic[D]


    def calculateProbability(self,x,W):
        numLabels     = len(self.idxToLabel.keys())+2
        numYs         = numLabels ** self.yNgramLen
        numPos        = len(self.idxToPos.keys())+2
        numXs         = numPos ** self.xNgramLen
        startX        = numPos-1
        startY        = numLabels-1
        l             = np.where(x==0)[0][0]#length of the sentence
        xCopy         = (x[:l]).astype(np.int)
        y             = np.ones(l-1).astype(np.int)
        A             = xCopy / (numPos*numYs)
        A             = A*numPos*numYs
        B             = (xCopy-A)/numYs
        B             = B*numYs
        AB            = A+B
        C             = np.zeros(l).astype(np.int)
        C[0]          = startY
        C[1:]         = y[:]
        D             = np.zeros(l).astype(np.int)
        D[-1]         = 0
        D[:-1]        = y[:]
        P             = np.zeros((numLabels-2)**(l-1))
        Y             = np.zeros([(numLabels-2)**(l-1),x.shape[0]]).astype(np.int)
        #calculate all possible tags
        ysFinished    = False
        i = 0
        while not ysFinished:
#        for i in range(l**(numLabels-2)):
        #for each tag, calculate the feature functions
            FF         =  AB + C*numLabels + D
            F,Fidx     =  self.nonZeroFeatFuncs(FF,W)
            P[i]       =  np.exp((W[Fidx]*F).sum())
            Y[i,:l-1]  = y[:]
            #for each tag, find the unscaled probability
            cnt        = l-2
            finished   = False
            while not finished:
                if cnt<0:
                    ysFinished = True
                    break
                y[cnt]      += 1
                C[cnt+1]    += 1
                D[cnt]      += 1
                if y[cnt] == numLabels-1:
                    y[cnt]   =  1
                    C[cnt+1]   =  1
                    D[cnt] =  1
                    cnt      -= 1
                else:
                    finished = True
            i += 1
        #normalize P
        P = P/P.sum()
        return Y,P
        #return the scaled probability along with different tags







