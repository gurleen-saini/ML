import numpy as np
import csv
from math import log
import operator
import pickle

# with open('drug200.csv','r') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# data_array = np.array(data)
# data_array = data_array[1:,:]
# print(data_array)

def createdataset():
    with open('drug200.csv','r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data)
    labels = list(data_array[0,:])  # Ensure labels are a list
    data_array = data_array[1:,:]
    return data_array,labels

data,labels = createdataset()
data,labels1 = createdataset()
print(data)
print(labels)

def calculateentropy(dataset):
    rows = len(dataset)
    labelscount={}
    for entry in dataset:
        currentlabel = entry[-1]
        if currentlabel not in labelscount:
            labelscount[currentlabel]=0
        labelscount[currentlabel]+=1
    entropy = 0.0
    for key in labelscount:
        probability = float(labelscount[key])/rows
        entropy -= probability*log(probability,2)
    return entropy

entropy = calculateentropy(data)
print(entropy)

def splitdataset(dataset,feature,value):
    resultingdataset = []
    for entry in dataset:
        if entry[feature]==value:
            mediatedataset = list(entry[:feature])
            mediatedataset.extend(list(entry[feature+1:]))
            resultingdataset.append(mediatedataset)
    return resultingdataset

dataset_m = splitdataset(data,1,'M')
dataset_f = splitdataset(data,1,'F')
print(len(data))
print(len(dataset_m))
print(len(dataset_f))
print(dataset_m)
print(dataset_f)

def choosingbestfeature(data):
    numfeatures = len(data[0])-1
    bestinfogain = 0.0
    bestfeature = -1
    baseentropy = calculateentropy(data)
    for i in range(numfeatures):
        feature_list = [example[i] for example in data]
        uniquevalue = set(feature_list)
        newentropy = 0.0
        for value in uniquevalue:
            subsetdataset = splitdataset(data,i,value)
            probability = float(len(subsetdataset))/float(len(data))
            newentropy += probability * calculateentropy(subsetdataset)
        infogain = baseentropy-newentropy
        if(infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature = i
    return bestfeature

bestfeaturetospliton = choosingbestfeature(data)
print(bestfeaturetospliton)
print(labels[bestfeaturetospliton])

def majoritycount(classlist):
    classcount = {}
    for label in classlist:
        if label not in classcount.keys():
            classcount[label] = 0
        classcount[label] +=1
    sortedclasscount = sorted(classcount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

majoritylabel = majoritycount(example[-1] for example in data)
print(majoritylabel)
    
def createtree(dataset,labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0])==1:
        return majoritycount(classlist)
    bestfeature = choosingbestfeature(dataset)
    bestfeaturelabel = labels[bestfeature]
    mytree = {bestfeaturelabel:{}}
    del(labels[bestfeature])
    featurevalues = [example[bestfeature] for example in dataset]
    uniquevalue = set(featurevalues)
    for value in uniquevalue:
        sublabels = labels[:]
        subdataset = splitdataset(dataset, bestfeature, value)
        mytree[bestfeaturelabel][value] = createtree(subdataset, sublabels)
    return mytree

mytree = createtree(data,labels)
print(mytree)

import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")             
arrow_args = dict(arrowstyle="<-")   

def plotNode(nodeTxt, centerPt, parentPt, nodeType): 
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', 
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getnumberleafs(tree):
    numleafs = 0
    firststr = list(tree.keys())[0]  # Convert keys to a list and get the first key
    seconddict = tree[firststr]
    for keys in seconddict.keys():
        if type(seconddict[keys]).__name__ == 'dict':
            numleafs += getnumberleafs(seconddict[keys])
        else:
            numleafs += 1
    return numleafs

def getTreeDepth(tree):
    maxDepth = 0
    firstStr = list(tree.keys())[0]  # Convert keys to a list and get the first key
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

leaf = getnumberleafs(mytree)
depth = getTreeDepth(mytree)
print(leaf)
print(depth)

def plotMidText(cntrPt, parentPt, txtString):     
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]    
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getnumberleafs(myTree)         
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getnumberleafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# createPlot(mytree)

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]

    # Find the index of the current feature label in featLabels
    featIndex = featLabels.index(firstStr)
    # print(featIndex)
    # print(secondDict)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # Recursively classify if the value is a subtree
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # If the value is a class label, return it
                classLabel = secondDict[key]
    return classLabel

ans = classify(mytree,labels1,['47','M','LOW','HIGH','10.114'])
print(ans)
ans = classify(mytree,labels1,['16','M','LOW','HIGH','12.006'])
print(ans)
ans = classify(mytree,labels1,['37','M','LOW','HIGH','12.006'])
print(ans)
ans = classify(mytree,labels1,['23','F','HIGH','HIGH','25.355'])
print(ans)
ans = classify(mytree,labels1,['24','M','HIGH','NORMAL','9.475'])
print(ans)
ans = classify(mytree,labels1,['60','F','HIGH','HIGH','13.303'])
print(ans)

# import pickle

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:  # Open the file in binary mode
        pickle.dump(inputTree, fw)


def grabTree(filename):
    with open(filename, 'rb') as fr:  # Open the file in binary read mode
        return pickle.load(fr)


storeTree(mytree,'store.txt')
storedtree = grabTree('store.txt')
print(storedtree)