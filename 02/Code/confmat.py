import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def confMat(Y,yPredicted,idxToLabel):
    keys      = idxToLabel.keys()
    numLabels = len(keys)
    confMat   = dict()
    for k in keys:
        confMat[k] = dict()
        idx        = np.where(Y==k)#true label
        sum        = 0
        for k2 in keys:
            idx2 = np.where(yPredicted[idx]==k2)
            sum  += idx2[0].shape[0]
            confMat[k][k2] = idx2[0].shape[0]
        sum = idx[0].shape[0]
        if sum==0:
            sum = 1.
        for k2 in keys:
            confMat[k][k2] /= float(sum)

    conf = np.zeros([numLabels,numLabels])
    for k in confMat.keys():
        for k2 in confMat[k].keys():
            conf[k-1,k2-1] = confMat[k][k2]

    return (conf*1000).astype(np.int)


def plotConfMat(confMat):
    fig = plt.figure(0)
    ax  = plt.gca()
    res = ax.imshow(confMat, cmap=cm.jet, interpolation='nearest')
    ax.get_yaxis().set_tick_params(direction='out')
#    for i, cas in enumerate(confMat):
#        for j, c in enumerate(cas):
#            if c>0:
#                plt.text(j-.2, i+.2, c, fontsize=14)
    cb = fig.colorbar(res)
    plt.show()