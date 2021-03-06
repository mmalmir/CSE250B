import pickle
import numpy as np
import pprint
from PuncData import PuncData
from PosTagFeats import PosTagFeats
from CRFClassifier import CRFClassifier
from confmat import *

######################
#### loading data ####
######################
print "loading data..."

l = PuncData()
X_train,y_train,X_test,y_test = l.load()
n = -1
ntest = -1
start = 0
X_train = X_train[start:start+n,:]
y_train = y_train[start:start+n,:]
######################
#### feat extrac. ####
######################

#f = open("puncdata.pickle","wb")
#pickle.dump((l,X_train,y_train,X_test,y_test),f)
#f.close()

#f = open("puncdata.pickle","rb")
#l,X_train,y_train,X_test,y_test = pickle.load(f)
#f.close()

print l.idxToYlabel
#
#print l.idxToPos

x_ngram_len = 2
y_ngram_len = 2

print "extracting features..."
ptf = PosTagFeats(idx_to_label = l.idxToYlabel,idx_to_pos=l.idxToPos,x_ngram_len=x_ngram_len,y_ngram_len=y_ngram_len)
X_train=ptf.transform(X_train,y_train)
X_test=ptf.transform(X_test,y_test)



############################################################
### TRAINING CONSTRASTIVE DIVERGENCE WITH MODEL AVERAGING ##
############################################################
if True:
    clf = dict()
    for i in range(1,7):
        print "training classifier on ",l.idxToYlabel[i]
        clf[i] = CRFClassifier(idx_to_label = l.idxToYlabel,idx_to_pos=l.idxToPos,
                                x_ngram_len=x_ngram_len,y_ngram_len=y_ngram_len,
        #                        train_method="CollinPerceptron",
                                train_method="CD",
                                sampling="random",
                                turn = i,
                                )
        clf[i].fit(X_train,y_train)
        idxNonZero = np.where(X_train!=0)

        Ypredicted = clf[i].transform(X_train)
        pCorrect = (y_train[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])

        print "################################################"
        print "TRAIN STATS:"
        print "################################################"
        for x in l.idxToYlabel.keys():
            idx = np.where(y_train==x)
            pCorrect = (y_train[idx]==Ypredicted[idx]).sum()/float(Ypredicted[idx].shape[0])
            print "for tag:",l.idxToYlabel[x]," rate is:",pCorrect, " num is:",Ypredicted[idx].shape[0]
        pCorrect = (y_train[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])
        print "train rate:",pCorrect

        print "\n"


    weight = dict()
    for i in range(1,7):
        idx = np.where(y_train==i)
        weight[i] = 1. / (idx[0].shape[0]+1.)

    #average weights
    W = np.zeros(clf[1].W.shape)
    for i in range(1,7):
        W += clf[i].W * weight[i]
#        for j in range(clf[0].W.shape[0]):
    #        clf[0].W[j] = max(clf[0].W[j] , clf[i].W[j])
    #clf[0].W = clf[0].W / len(clf)
    clf[1].W = W
    clf = clf[1]



else:
    clf= CRFClassifier(idx_to_label = l.idxToYlabel,idx_to_pos=l.idxToPos,
                             x_ngram_len=x_ngram_len,y_ngram_len=y_ngram_len,
                             #                        train_method="CollinPerceptron",
                             train_method="CD",
                             sampling="posterior",
                             turn = -1,
                             )
    clf.fit(X_train,y_train)


idxNonZero = np.where(X_train!=0)

Ypredicted = clf.transform(X_train)
pCorrect = (y_train[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])

conf = confMat(y_train,Ypredicted,l.idxToYlabel)
pprint.pprint(conf)
labels = [l.idxToYlabel[k] for k in  range(1,7)]
print labels
plotConfMat(conf,labels,"confMatTrain.png")


print "################################################"
print "TRAIN STATS:"
print "################################################"
for x in l.idxToYlabel.keys():
    idx = np.where(y_train==x)
    pCorrect = (y_train[idx]==Ypredicted[idx]).sum()/float(Ypredicted[idx].shape[0])
    print "for tag:",l.idxToYlabel[x]," rate is:",pCorrect, " num is:",Ypredicted[idx].shape[0]
pCorrect = (y_train[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])
print "train rate:",pCorrect

print "\n"

print "################################################"
print "TEST STATS:"
print "################################################"


start = 0
X_test = X_test[start:start+ntest,:]
y_test = y_test[start:start+ntest,:]

idxNonZero = np.where(X_test!=0)

Ypredicted = clf.transform(X_test)
pCorrect = (y_test[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])

#sentence level recognition rates
correct    = np.any(y_test!=Ypredicted,axis=1)
print "sentence level recognition for test set:",1. - correct.sum()/float(correct.shape[0])


for x in l.idxToYlabel.keys():
    idx = np.where(y_test==x)
    pCorrect = (y_test[idx]==Ypredicted[idx]).sum()/float(Ypredicted[idx].shape[0])
    print "for tag:",l.idxToYlabel[x]," rate is:",pCorrect, " num is:",Ypredicted[idx].shape[0]
pCorrect = (y_test[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])
print "test rate:",pCorrect


conf = confMat(y_test,Ypredicted,l.idxToYlabel)
pprint.pprint(conf)
labels = [l.idxToYlabel[k] for k in  range(1,7)]
print labels
plotConfMat(conf,labels,"confMatTest.png")



##### testing numeric features
"""

map = l.idxToYlabel
map[-1] = ""
for i in range(y_train.shape[0]):
    for k in range(y_train.shape[1]):
        if y_train[i,k]>0:
            if map[y_train[i,k]]!=l.trainLabels[i][k]:
                print map[y_train[i,k]],l.trainLabels[i][k]
        if X_train[i,k]>0:
            if l.idxToPos[X_train[i,k]]!=l.trainPos[i][k]:
                print l.idxToPos[X_train[i,k]],l.trainPos[i][k]


for i in range(y_test.shape[0]):
    for k in range(y_test.shape[1]):
        if y_test[i,k]>0:
            if map[y_test[i,k]]!=l.testLabels[i][k]:
                print map[y_test[i,k]],l.testLabels[i][k]
        if X_test[i,k]>0:
            if l.idxToPos[X_test[i,k]]!=l.testPos[i][k]:
                print l.idxToPos[X_test[i,k]],l.testPos[i][k]
"""