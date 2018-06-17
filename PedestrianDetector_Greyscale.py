from PIL import Image
import math
import glob
import cv2
import os
import random
from numpy.linalg import inv
import numpy as np
from matplotlib import pyplot
import pylab as pl
from matplotlib import pyplot
import pylab as pl
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\TrainingNonPed1\\*.bmp
#
bestPixels=[97,98,99,100,101,129,130,277,351,386,387,389,507,524,560,632,640,647]
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\TrainingNonPed1\\*.bmp
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\TrainingPed\\*.bmp
def createDataset():
    pixelListNP1=[]
    pixelListNP2=[]
    pixelListP=[]
    print('eachImage1')
    for eachImage1 in glob.glob("D:\\PG\\Sem3\\IPPR\\LAB\\DaimlerBenchmark\\TrainingData\\Pedestrians\\18x36\\*.bmp"):
        pixelsNP1=Image.open(str(eachImage1),'r')
        pixelListNP1.append(list(pixelsNP1.getdata()))
    nonPedData=np.array(pixelListNP1)
    print('eachImage3')
    for eachImage3 in glob.glob("D:\\PG\\Sem3\\IPPR\\LAB\\DaimlerBenchmark\\TrainingData\\NonPedestrians\\18x36\\1_non-ped_examples bmp\\*.bmp"):
        pixelsP=Image.open(str(eachImage3),'r')
        pixelListP.append(list(pixelsP.getdata()))
    pedData= np.array(pixelListP)

    return nonPedData, pedData

#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\TestingNonPed\\*.bmp
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\TestingPed\\*.bmp
#"C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestNonPed\\*.bmp"
#"C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestPed\\*.bmp"
def createTestDataset():
    pixelListNP1=[]
    pixelListNP2=[]
    pixelListP=[]
    for eachImage1 in glob.glob("D:\\PG\\Sem3\\IPPR\\LAB\\DaimlerBenchmark\\TrainingData\\Pedestrians\\18x36\\*.bmp"):
        pixelsNP1=Image.open(str(eachImage1),'r')
        pixelListNP1.append(list(pixelsNP1.getdata()))
    nonPedTestData=np.array(pixelListNP1)

    for eachImage3 in glob.glob("D:\\PG\\Sem3\\IPPR\\LAB\\DaimlerBenchmark\\TrainingData\\NonPedestrians\\18x36\\*.bmp"):
        pixelsP=Image.open(str(eachImage3),'r')
        pixelListP.append(list(pixelsP.getdata()))
    pedTestData= np.array(pixelListP)

    return nonPedTestData, pedTestData


def rowWiseBestPixelMatrix(bestPixels,data):
    bestPixelMatrix=[]
    for eachBestPixel in bestPixels:
        print(eachBestPixel)
        bestPixelMatrix.append(data[eachBestPixel])
    return np.array(bestPixelMatrix)


def mean(inputSeq):
    meanList=[]
    for eachList in inputSeq:
        tempAvrg=[float(sum(eachList))/len(eachList)]
        meanList.append(tempAvrg)
    return np.array(meanList)

def covarianceMatrix(inputSeq):
    return np.cov(inputSeq)

def matrixDeter(inputArray):
    return np.linalg.det(inputArray)


def discriminantFun(inputDeter,inputPixelSequence,meanList,covMatrix):


    tempa=(inputPixelSequence-meanList).transpose()
    tempb=np.dot(tempa,inv(covMatrix))
    tempc=np.dot(tempb,(inputPixelSequence-meanList))

    discValue= math.log(0.5)-((0.5)*math.log(inputDeter))-(np.dot(tempc,0.5))
    return discValue



def bestImagePixel(bestPixels,eachImageSample):
    tempList=[]
    bestPix=[]
    for eachBestPixel in bestPixels:
        templist=eachImageSample[eachBestPixel]
        bestPix.append([templist])
    return np.array(bestPix)



def classClassify(inputSequence,bestPixels,inputDeterP,meanListP,covMatrixP,threshold):
    totalSamples=float(len(inputSequence))
    pedDetected=0
    nonPedDetected=0

    for eachImageSample in inputSequence:
        bestPixelFromEachImage=bestImagePixel(bestPixels,eachImageSample)
        dP=discriminantFun(inputDeterP,bestPixelFromEachImage,meanListP,covMatrixP)

        if (dP[0][0])>threshold:
            pedDetected+=1
        else:
            nonPedDetected+=1
    return pedDetected, nonPedDetected,totalSamples



def main():
    nonPedData, pedData=createDataset()
    print(nonPedData)
    print(pedData)
    nonPedTestData, pedTestData=createTestDataset()
    print(nonPedTestData)
    print(pedTestData)
    transposedPedData=pedData.transpose()
    print(transposedPedData)
    pedBestPixelMatrix= rowWiseBestPixelMatrix(bestPixels,transposedPedData)
    pedBestPixelMatrixMean=mean(pedBestPixelMatrix)
    pedBestPixelMatrixCovariance=covarianceMatrix(pedBestPixelMatrix)
    pedBestPixelMatrixCovarDeterminant=matrixDeter(pedBestPixelMatrixCovariance)


    dMax=discriminantFun(pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance)
    dkMax= dMax[0][0]
    print(dkMax)
    thresholdRange=range(1,21)
    thresholdValues=[(20*dkMax)/float(x) for x in thresholdRange]


    pedDetectedP,nonPedDetectedP,totalSamplesP=classClassify(pedTestData,bestPixels,pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance,thresholdValues[16])
    pedDetectedNP,nonPedDetectedNP,totalSamplesNP=classClassify(nonPedTestData,bestPixels,pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance,thresholdValues[16])
    #print thresholdValues[16]

    classifierAccuracy=((pedDetectedP+nonPedDetectedNP)/float(totalSamplesP+totalSamplesNP))*100


    ped1Entry=pedDetectedP/float(totalSamplesP)
    nonPed1Entry=nonPedDetectedP/float(totalSamplesP)
    ped2Entry=pedDetectedNP/float(totalSamplesNP)
    nonPed2Entry=nonPedDetectedNP/float(totalSamplesNP)
    confMatrix=[[ped1Entry,nonPed1Entry],[ped2Entry,nonPed2Entry]]
    trainDataConfusionMatrix= np.dot(confMatrix,100)
    print("\n")
    print("The threshold with highest accuracy:  %d" % thresholdValues[16])
    print("\n")
    print("Total number of pedestrian samples: %d" % totalSamplesP)
    print("\n")
    print("Correctly detected pedestrian Samples [True postives]: %d" % pedDetectedP)
    print("Wrongly detected pedestrian Samples [False negatives]: %d" % nonPedDetectedP)
    print("\n")
    print("Total number of non-pedestrian samples: %d" % totalSamplesNP)
    print("\n")
    print("Wrongly detected non-pedestrian Samples [False positives]: %d" % pedDetectedNP)
    print("Correctly detected non-pedestrian Samples [True nagatives]: %d" % nonPedDetectedNP)
    print("\n")
    print("The greyscale based Pedestrian Detector confusion matrix: ")
    print(trainDataConfusionMatrix)
    print("\n")
    print("The greyscale based Pedestrian Detector accuracy: %f"  % classifierAccuracy)
    print("\n")

main()
