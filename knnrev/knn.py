from numpy import *
import numpy as np
from numpy import tile
import operator

# operated on first 8 eatures only

def createdataset():
    data = [
    [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471],
    [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017],
    [19.69, 21.25, 130, 1203, 0.1096, 0.1599, 0.1974, 0.1279],
    [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052],
    [20.29, 14.34, 135.1, 1297, 0.1003, 0.1328, 0.198, 0.1043],
    [12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089],
    [18.25, 19.98, 119.6, 1040, 0.09463, 0.109, 0.1127, 0.074],
    [13.71, 20.83, 90.2, 577.9, 0.1189, 0.1645, 0.09366, 0.05985],
    [13, 21.82, 87.5, 519.8, 0.1273, 0.1932, 0.1859, 0.09353],
    [12.46, 24.04, 83.97, 475.9, 0.1186, 0.2396, 0.2273, 0.08543],
    [16.02, 23.24, 102.7, 797.8, 0.08206, 0.06669, 0.03299, 0.03323],
    [15.78, 17.89, 103.6, 781, 0.0971, 0.1292, 0.09954, 0.06606],
    [19.17, 24.8, 132.4, 1123, 0.0974, 0.2458, 0.2065, 0.1118],
    [15.85, 23.95, 103.7, 782.7, 0.08401, 0.1002, 0.09938, 0.05364],
    [13.73, 22.61, 93.6, 578.3, 0.1131, 0.2293, 0.2128, 0.08025],
    [14.54, 27.54, 96.73, 658.8, 0.1139, 0.1595, 0.1639, 0.07364],
    [14.68, 20.13, 94.74, 684.5, 0.09867, 0.072, 0.07395, 0.05259],
    [16.13, 20.68, 108.1, 798.8, 0.117, 0.2022, 0.1722, 0.1028],
    [19.81, 22.15, 130.0, 1260.0, 0.09831, 0.1027, 0.1479, 0.09498],
    [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781],
    [13.08, 15.71, 85.63, 520.0, 0.1075, 0.127, 0.04568, 0.0311],
    [9.504, 12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076],
    [15.34, 14.26, 102.5, 704.4, 0.1073, 0.2135, 0.2077, 0.09756],
    [21.16, 23.04, 137.2, 1404.0, 0.09428, 0.1022, 0.1097, 0.08632],
    [16.65, 21.38, 110.0, 904.6, 0.1121, 0.1457, 0.1525, 0.0917],
    [17.14, 16.4, 116.0, 912.7, 0.1186, 0.2276, 0.2229, 0.1401],
    [14.58, 21.53, 97.41, 644.8, 0.1054, 0.1868, 0.1425, 0.08783],
    [18.61, 20.25, 122.1, 1094, 0.0944, 0.1066, 0.149, 0.07731],
    [15.3, 25.27, 102.4, 732.4, 0.1082, 0.1697, 0.1683, 0.08751],
    [17.57, 15.05, 115, 955.1, 0.09847, 0.1157, 0.09875, 0.07953],
    [18.63, 25.11, 124.8, 1088, 0.1064, 0.1887, 0.2319, 0.1244],
    [11.84, 18.7, 77.93, 440.6, 0.1109, 0.1516, 0.1218, 0.05182],
    [17.02, 23.98, 112.8, 899.3, 0.1197, 0.1496, 0.2417, 0.1203],
    [19.27, 26.47, 127.9, 1162, 0.09401, 0.1719, 0.1657, 0.07593],
    [16.13, 17.88, 107, 807.2, 0.104, 0.1559, 0.1354, 0.07752],
    [16.74, 21.59, 110.1, 869.5, 0.0961, 0.1336, 0.1348, 0.06018],
    [14.25, 21.72, 93.63, 633, 0.09823, 0.1098, 0.1319, 0.05598],
    [13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923],
    [14.99, 25.2, 95.54, 698.8, 0.09387, 0.05131, 0.02398, 0.02899]]

    labels=[    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "B",    "B",    "B",    "M",    "M",    "M",    "M",    "M",    "M",    "M",  "M",    "M",    "M",    "M",    "M",    "M",    "M",    "M",    "B",    "M"]

    return data,labels

data,labels = createdataset()
print(len(data))
print(len(labels))

def classify0(tosearch,dataset,labels,k):
    dataset = np.array(dataset)
    datasize = dataset.shape[0]
    newmat = tile(tosearch,(datasize,1)) - dataset
    sqmat = newmat**2
    summat = sqmat.sum(axis=1)
    resmat = summat**0.5
    sortedmat = resmat.argsort()
    labellist={}
    for i in range(k):
        labellist[labels[sortedmat[i]]] = labellist.get(labels[sortedmat[i]], 0) + 1
    
    sortedmatrixresult = sorted(labellist.items(),key=operator.itemgetter(1),reverse=True)
    
    return sortedmatrixresult[0][0]

numbers_1 =[
    13.03,
    18.42,
    82.61,
    523.8,
    0.08983,
    0.03766,
    0.02562,
    0.02923
]
numbers_2 =  [
    13.73,
    22.61,
    93.6,
    578.3,
    0.1131,
    0.2293,
    0.2128,
    0.08025
]

a = classify0(numbers_1,data,labels,3)
b = classify0(numbers_2,data,labels,3)

print(a)
print(b)

def normalizing(dataset):
    dataset = np.array(dataset)
    minvalue = dataset.min(0)
    maxvalue = dataset.max(0)
    rangevalue = maxvalue-minvalue
    resultmat = zeros(shape(dataset))
    m = dataset.shape[0]
    resultmat = dataset - tile(minvalue,(m,1))
    resultmat = resultmat/tile(rangevalue,(m,1))
    return resultmat,rangevalue,minvalue

def classifiertest():
    testratio = 0.1
    data,labels = createdataset()
    normalizedmat,rangevalue,minvalue = normalizing(data)
    m = normalizedmat.shape[0]
    indexvalue = int(m*testratio)
    errorvalue = 0.0
    for i in range(indexvalue):
        classifierresult = classify0(normalizedmat[i,:],normalizedmat[indexvalue:m,:],labels[indexvalue:m],3)
        print("classifierresult is %c and actual result is %c" % (classifierresult,labels[i]))
        if classifierresult!=labels[i]:
            errorvalue+=1
    print("Error rate is %d" % (errorvalue/float(indexvalue)))

def classifypositiveornot():
    radius_mean = float(input("radius_mean "))
    texture_mean = float(input("texture_mean "))
    perimeter_mean = float(input("perimeter_mean "))
    area_mean = float(input("area_mean "))
    smoothness_mean = float(input("smoothness_mean "))
    compactness_mean = float(input("compactness_mean "))
    concavity_mean = float(input("concavity_mean "))
    concave_points_mean = float(input("concave_points_mean "))
    data,labels = createdataset()
    normalizedmat,rangevalue,minvalue = normalizing(data)
    inputarray = array([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean])
    classifierresult = classify0((inputarray-minvalue)/rangevalue,normalizedmat,labels,3)
    if classifierresult=='M':
        print("positive")
    elif classifierresult=='B':
        print("negative")

normalizedmat,rangevalue,minvalue = normalizing(data)
print(normalizedmat)
print(rangevalue)
print(minvalue)

classifiertest()
classifypositiveornot()