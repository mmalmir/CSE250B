import pickle
from PuncData import PuncData
from PosTagFeats import PosTagFeats
from CRFClassifier import CRFClassifier

######################
#### loading data ####
######################
print "loading data..."

l = PuncData()
X_train,y_train,X_test,y_test = l.load()
n = 500
start = 100
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
#                        train_method="CollinPerceptron")
                        train_method="CD")
clf.fit(X_train,y_train)






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