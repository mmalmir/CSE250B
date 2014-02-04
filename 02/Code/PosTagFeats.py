import re
import os
import string
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
        numLabels = len(self.idxToLabel.keys())
        numPos    = len(self.idxToPos)
        print "number of f functions:",numLabels**self.yNgramLen*numPos**self.xNgramLen
        print numLabels,numPos


