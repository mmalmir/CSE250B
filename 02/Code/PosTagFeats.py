import re
import os
import string
import copy
import numpy as np
from topia.termextract import tag
from itertools import chain
from transformer import Transformer


class PosTagFeats(Transformer):
    
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

    def get_params(self,deep=False):
        return dict({"x_ngram_len"  :self.xNgramLen,
                      "y_ngram_len" :self.yNgramLen,
                      "idx_to_label":self.idxToLabel,
                      "idx_to_pos"  :self.idxToPos,
                    })

    def transform(self,X,y=None,**kwargs):
        self.set_params(**kwargs)
        numLabels      = len(self.idxToLabel.keys())+2
        numLabelsSq    = numLabels**2
        numPos         = len(self.idxToPos.keys())+2
        n,d            = X.shape
        endIdx         = X==-1
        XCopy          = copy.deepcopy(X)+1
        YCopy          = copy.deepcopy(y)+1
        XCopy,YCopy    = XCopy.astype(np.int),YCopy.astype(np.int)
        startX         = XCopy.max()+1
        startY         = YCopy.max()+1
        X_transformed  = np.zeros([n,d]).astype(np.int)
        # X:2-grams and Y:2-grams
        if self.xNgramLen==2 and self.yNgramLen==2:
            for i in range(d-1):
                X_transformed[:,i+1] = XCopy[:,i]*numPos*numLabelsSq + XCopy[:,i+1]*numLabelsSq + YCopy[:,i]*numLabels + YCopy[:,i+1]
            X_transformed[:,0] = startX*numPos*numLabelsSq + XCopy[:,1]*numLabelsSq + startY*numLabels + YCopy[:,1]
        
        # X:1-grams and Y:2-grams
        elif self.xNgramLen==1 and self.yNgramLen==2:
            for i in range(d-1):
                X_transformed[:,i+1] = XCopy[:,i+1]*numLabelsSq + YCopy[:,i]*numLabels + YCopy[:,i+1]
            X_transformed[:,0] = XCopy[:,0]*numLabelsSq + startY*numLabels + YCopy[:,1]
        
        # X:2-grams and Y:1-grams
        elif self.xNgramLen==2 and self.yNgramLen==1:
            for i in range(d-1):
                X_transformed[:,i+1] = XCopy[:,i]*numPos*numLabels + XCopy[:,i+1]*numLabels + YCopy[:,i+1]
            X_transformed[:,0] = startX*numPos*numLabels + XCopy[:,1]*numLabels + YCopy[:,1]
        
        # X:1-grams and Y:1-grams
        elif self.xNgramLen==2 and self.yNgramLen==1:
            for i in range(d):
                X_transformed[:,i] = XCopy[:,i]*numPos + YCopy[:,i]

        X_transformed[endIdx] = -1
        return X_transformed


        ####### TEST indexing ######
"""
        for i in range(1,d):
            A = X_transformed[:,i] / (numPos*numLabelsSq)
            B = (X_transformed[:,i] - A*numPos*numLabelsSq) / numLabelsSq
            C = (X_transformed[:,i] - A*numPos*numLabelsSq - B*numLabelsSq) / numLabels
            D = (X_transformed[:,i] - A*numPos*numLabelsSq - B*numLabelsSq) % numLabels
            if np.any(A!=XCopy[:,i-1]): #or np.any(B!=XCopy[:,i]) or np.any(C!=YCopy[:,i-1]) or np.any(D!=YCopy[:,i]):
                idx = np.where(A!=XCopy[:,i-1])
                print A[idx],XCopy[idx,i-1]
            if np.any(B!=XCopy[:,i]): #or np.any(B!=XCopy[:,i]) or np.any(C!=YCopy[:,i-1]) or np.any(D!=YCopy[:,i]):
                idx = np.where(B!=XCopy[:,i])
                print B[idx],XCopy[idx,i]
            if np.any(C!=YCopy[:,i-1]): #or np.any(B!=XCopy[:,i]) or np.any(C!=YCopy[:,i-1]) or np.any(D!=YCopy[:,i]):
                idx = np.where(C!=YCopy[:,i-1])
                print C[idx],YCopy[idx,i-1]
            if np.any(D!=YCopy[:,i]): #or np.any(B!=XCopy[:,i]) or np.any(C!=YCopy[:,i-1]) or np.any(D!=YCopy[:,i]):
                idx = np.where(D!=YCopy[:,i])
                print D[idx],YCopy[idx,i]

"""
#        print X_transformed







