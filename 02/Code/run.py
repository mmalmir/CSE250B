from PuncData import PuncData

######################
#### loading data ####
######################

l = PuncData()
X_train,y_train,X_test,y_test = l.load()

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