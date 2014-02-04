import re
import os
import string
import numpy as np
from topia.termextract import tag
from itertools import chain

class PosTagFeats(Transformer):
    
    def __init__(self,**kwargs):
        self.set_params(**kwargs)

    
    def set_params(self,**kwargs):
        for k in kwargs:
            if k=="path":
                self.dataPath = kwargs[k]


