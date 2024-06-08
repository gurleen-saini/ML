import numpy as np
from numpy import *
import csv
import re

def createdataset():
    with open('Emotion_classify_Data.csv','r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data)
    data_array = data_array[1:,:]
    labels = list(data_array[:,1])  # Ensure labels are a list
    data_array = data_array[:,0]
    return data_array,labels

data,labels = createdataset()
# print(data)
# print(labels)

# fear=0
# joy=1
# anger=2

for i in range(len(labels)):
    if labels[i] == 'fear':
        labels[i]=2
    elif labels[i] == 'joy':
        labels[i]=1
    else:
        labels[i]=0

# print(labels)

def textParse(bigString):
    listOfTokens = re.findall(r'\b\w+\b', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

dataset = [textParse(text) for text in data]

# print(dataset)

def createVocabList(dataSet):
    vocabSet = set([])                         
    for document in dataSet:
        vocabSet = vocabSet | set(document)          
    return list(vocabSet)

vocablist = createVocabList(dataset)
# print(vocablist)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)             
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

trainmatrix = []
for posts in dataset:
    trainmatrix.append(setOfWords2Vec(vocablist,posts))

# print(trainmatrix)


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    sumfear = 0.0
    for i in trainCategory:
        if i == 2:
            sumfear += 1

    pfear = sumfear/float(numTrainDocs)

    sumjoy = 0.0
    for i in trainCategory:
        if i == 1:
            sumjoy += 1

    pjoy = sumjoy/float(numTrainDocs)

    p0Num = ones(numWords); p1Num = ones(numWords); p2Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0; p2Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]            
            p1Denom += sum(trainMatrix[i])            
        elif trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        else:
            p2Num += trainMatrix[i]
            p2Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    p2Vect = log(p2Num/p2Denom)
    return p0Vect,p1Vect,p2Vect,pjoy,pfear

p0prob,p1prob,p2prob,probjoy,probfear = trainNB0(trainmatrix,labels)
# print(p0prob)
# print(p1prob)
# print(p2prob)
print(probjoy)
print(probfear)

def classifyNB(vec2Classify, p0Vec, p1Vec,p2Vec, pClass1,pClass2):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)         
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1 - pClass2)
    p2 = sum(vec2Classify * p2Vec) + log(pClass2)
    if (p1 > p0)&(p1 >p2):
        return 1
    elif (p0 > p1)&(p0 > p2): 
        return 0
    else:
        return 2
    
testentry = ['love','cute']
thisDoc = array(setOfWords2Vec(vocablist, testentry))
ans = classifyNB(thisDoc,p0prob,p1prob,p2prob,probjoy,probfear)
if ans==0:
    ans = "Angry"
elif ans==1:
    ans = "Joy"
else:
    ans = "Fear"

print (testentry,'classified as: ',ans)

testentry = ['terrified']
thisDoc = array(setOfWords2Vec(vocablist, testentry))
ans = classifyNB(thisDoc,p0prob,p1prob,p2prob,probjoy,probfear)
if ans==0:
    ans = "Angry"
elif ans==1:
    ans = "Joy"
else:
    ans = "Fear"

print (testentry,'classified as: ',ans)

testentry = ['irritable']
thisDoc = array(setOfWords2Vec(vocablist, testentry))
ans = classifyNB(thisDoc,p0prob,p1prob,p2prob,probjoy,probfear)
if ans==0:
    ans = "Angry"
elif ans==1:
    ans = "Joy"
else:
    ans = "Fear"

print (testentry,'classified as: ',ans)