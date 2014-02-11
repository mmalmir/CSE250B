import pickle
import numpy as np
from PuncData import PuncData
from PosTagFeats import PosTagFeats
from CRFClassifier import CRFClassifier

######################
#### loading data ####
######################
print "loading data..."

l = PuncData()
X_train,y_train,X_test,y_test = l.load()
n = 1000
ntest = 2000
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

#print l.idxToYlabel
#
#print l.idxToPos

x_ngram_len = 2
y_ngram_len = 2

print "extracting features..."
ptf = PosTagFeats(idx_to_label = l.idxToYlabel,idx_to_pos=l.idxToPos,x_ngram_len=x_ngram_len,y_ngram_len=y_ngram_len)
X_train=ptf.transform(X_train,y_train)
X_test=ptf.transform(X_test,y_test)


print "training classifier..."
clf = CRFClassifier(idx_to_label = l.idxToYlabel,idx_to_pos=l.idxToPos,
                        x_ngram_len=x_ngram_len,y_ngram_len=y_ngram_len,
#                        train_method="CollinPerceptron",
                        train_method="CD",
                        sampling="random",
                        )
#clf.turn = 5
#clf.fit(X_train,y_train)

#clf.turn = 1
#clf.fit(X_train,y_train)
#
clf.turn = 3
clf.fit(X_train,y_train)
#
#clf.turn = 6
#clf.fit(X_train,y_train)

#clf.train_method = "CD"
#clf.sampling="random"
#clf.fit(X_train,y_train)
#clf.sampling="posterior"
#clf.fit(X_train,y_train)
#clf.sampling="guided"
#clf.fit(X_train,y_train)

idxNonZero = np.where(X_train!=0)

Ypredicted = clf.transform(X_train)
pCorrect = (y_train[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])

print l.idxToYlabel

#hist = np.zeros(Ypredicted.shape[0])
#for i in range(Ypredicted.shape[0]):
#    idxEnd = np.where(Ypredicted[i,:]==0)[0][0]
#    hist[i] = Ypredicted[i,idxEnd-1]
#cnt,bins= np.histogram(hist,bins=np.arange(0,8))
#print hist
#print l.idxToYlabel
#print cnt,bins
#
#
#hist = np.zeros(y_train.shape[0])
#for i in range(y_train.shape[0]):
#    idxEnd = np.where(y_train[i,:]==0)[0][0]
#    hist[i] = y_train[i,idxEnd-1]
#print hist
#cnt,bins= np.histogram(hist,bins=np.arange(0,8))
#print l.idxToYlabel
#print cnt,bins


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


for x in l.idxToYlabel.keys():
    idx = np.where(y_test==x)
    pCorrect = (y_test[idx]==Ypredicted[idx]).sum()/float(Ypredicted[idx].shape[0])
    print "for tag:",l.idxToYlabel[x]," rate is:",pCorrect, " num is:",Ypredicted[idx].shape[0]
pCorrect = (y_test[idxNonZero]==Ypredicted[idxNonZero]).sum()/float(Ypredicted[idxNonZero].shape[0])
print "test rate:",pCorrect





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