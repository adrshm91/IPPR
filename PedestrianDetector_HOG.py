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
from skimage.feature import hog
from skimage import data, color, exposure
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTrainPed\\*.bmp
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTrainNonPed\\*.bmp
bestPixels=[11,12,16,23,29,30,32,33,42,43,48,55,59,75,80,87,94,111,118,128,136]
#bestPixels=range(108)
def createDataset():
    fdTestListP=[]
    fdTestListNP=[]
    for eachTestImage2 in glob.glob("C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTrainPed\\*.bmp"):
        imgTestP=cv2.imread(eachTestImage2,0)
        fdTestP, hog_imageTestP = hog(imgTestP, orientations=8, pixels_per_cell=(6,6),cells_per_block=(1, 1), visualise=True)
        fdTestListP.append(list(fdTestP))
    pedData= 10*(np.array(fdTestListP))
    for eachTestImage1 in glob.glob("C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTrainNonPed\\*.bmp"):
        imgTestNP=cv2.imread(eachTestImage1,0)
        fdTestNP, hog_imageTestNP = hog(imgTestNP, orientations=8, pixels_per_cell=(6,6),cells_per_block=(1, 1), visualise=True)
        fdTestListNP.append(list(fdTestNP))
    nonPedData=np.array(fdTestListNP)
    return pedData,nonPedData

#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestPed\\*.bmp
#C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestNonPed\\*.bmp

def createTestDataset():

    fdTestListP=[]
    fdTestListNP=[]
    for eachTestImage2 in glob.glob("C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestPed\\*.bmp"):
        imgTestP=cv2.imread(eachTestImage2,0)
        fdTestP, hog_imageTestP = hog(imgTestP, orientations=8, pixels_per_cell=(6,6),cells_per_block=(1, 1), visualise=True)
        fdTestListP.append(list(fdTestP))
    for eachTestImage1 in glob.glob("C:\\Users\\Ferin\\helloworld\\DaimlerBenchmark\\FinalTestNonPed\\*.bmp"):
        imgTestNP=cv2.imread(eachTestImage1,0)
        fdTestNP, hog_imageTestNP = hog(imgTestNP, orientations=8, pixels_per_cell=(6,6),cells_per_block=(1, 1), visualise=True)
        fdTestListNP.append(list(fdTestNP))
    nonPedTestData=10*(np.array(fdTestListNP))
    pedTestData=10*(np.array(fdTestListP))

    return pedTestData,nonPedTestData


def rowWiseBestPixelMatrix(bestPixels,data):
    bestPixelMatrix=[]
    for eachBestPixel in bestPixels:
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
    pedData,nonPedData=createDataset()
    pedTestData,nonPedTestData=createTestDataset()

    transposedPedData=pedData.transpose()
    pedBestPixelMatrix= rowWiseBestPixelMatrix(bestPixels,transposedPedData)
    pedBestPixelMatrixMean=mean(pedBestPixelMatrix)
    pedBestPixelMatrixCovariance=covarianceMatrix(pedBestPixelMatrix)
    pedBestPixelMatrixCovarDeterminant=matrixDeter(pedBestPixelMatrixCovariance)

    dMax=discriminantFun(pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance)
    dkMax= dMax[0][0]
    #print dkMax
    thresholdRange=range(1,101)
    thresholdValues=[(100*dkMax)/float(x) for x in thresholdRange]


    pedDetectedP,nonPedDetectedP,totalSamplesP=classClassify(pedTestData,bestPixels,pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance,thresholdValues[5])
    pedDetectedNP,nonPedDetectedNP,totalSamplesNP=classClassify(nonPedTestData,bestPixels,pedBestPixelMatrixCovarDeterminant,pedBestPixelMatrixMean,pedBestPixelMatrixCovariance,thresholdValues[5])

    #print thresholdValues[5]

    classifierAccuracy=((pedDetectedP+nonPedDetectedNP)/float(totalSamplesP+totalSamplesNP))*100


    ped1Entry=pedDetectedP/float(totalSamplesP)
    nonPed1Entry=nonPedDetectedP/float(totalSamplesP)
    ped2Entry=pedDetectedNP/float(totalSamplesNP)
    nonPed2Entry=nonPedDetectedNP/float(totalSamplesNP)
    confMatrix=[[ped1Entry,nonPed1Entry],[ped2Entry,nonPed2Entry]]
    trainDataConfusionMatrix= np.dot(confMatrix,100)
    print "The threshold with highest accuracy:  %d" % thresholdValues[5]
    print "\n"
    print "Total number of pedestrian samples: %d" % totalSamplesP
    print "\n"
    print "Correctly detected pedestrian Samples [True postives]: %d" % pedDetectedP
    print "Wrongly detected pedestrian Samples [False negatives]: %d" % nonPedDetectedP
    print "\n"
    print "Total number of non-pedestrian samples: %d" % totalSamplesNP
    print "\n"
    print "Wrongly detected non-pedestrian Samples [False positives]: %d" % pedDetectedNP
    print "Correctly detected non-pedestrian Samples [True nagatives]: %d" % nonPedDetectedNP
    print "\n"
    print "The HOG based Pedestrian Classifier confusion matrix: "
    print trainDataConfusionMatrix
    print "\n"
    print "The HOG based Pedestrian Classifier accuracy: %f"  % classifierAccuracy
    print "\n"

main()
