#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:22:15 2019

@author: xiangshuyang
"""


import numpy as np 
import math 
def classify(data,threshold,k,thresin):  #self-defined predictor 
    classifyarray = np.ones((np.shape(data)[0],1))
    if (thresin=='lf'):
        classfifyarray[data[:,k]<=threshold]=-1
    else :
         classifyarray[data[:,k]>threshold]=-1
    return classifyarray



def buildclassify(data,label,D,numstep):
    m,n=np.shape(data)
    minerror= 10**10
    bestStump = {}
    bestclass=np.ones((m,1))
    for i in range(n):
        rangemin=data[:,i].min()
        rangemax=data[:,i].max()
        rangei=(rangemax-rangemin)/numstep
        for j in range(-1,int(numstep)+1):
            for thresin in ['lt','gt']:
                threshold= rangemin+ j*rangei 
                predictarray= classify(data,threshold,i,thresin)
                errorarray=np.ones((m,1))
                errorarray[predictarray==label]=0 
                weighterror = D.T*errorarray
                if weighterror< minerror:
                    minerror=weighterror 
                    bestclass = predictarray
                    bestStump['Dim'] = i
                    bestStump['Thresh'] = threshold
                    bestStump['Thesin'] = thresin
    return bestclass, bestStump, minerror 



def Adaboost(data,label,numiter):
    m,n = np.shape(data)
    numstep= 2 
    weakclass=[]
    aggclass= np.zeros((m,1))
    D = np.matrix(np.ones((m,1))/m)
    aggclass = np.matrix(np.zeros((m,1)))
    for i in range(numiter): 
        data,label,D,numstep
        bestclass,bestStump,error = buildclassify(data,label,D,numstep)
        alpha = 1/2*math.log((1-error)/error)
        
        bestStump['alpha'] = alpha
        weakclass.append(bestStump)
        expon=np.exp(np.multiply(-alpha*label,bestclass))
        Dexpon= np.multiply(D,expon)
        D=Dexpon/Dexpon.sum()
        aggclass += alpha*bestclass
        aggerror = np.multiply(((aggclass!= label)).astype(int),np.ones((m,1)))
        errorate = aggerror.sum()/m
    return weakclass 

def adaclassify(data,classarra):
    m,n= np.shape(data)
    aggclass = np.matrix(np.zeros((m,1)))
    for i in range(len(classarra)):
        classest = classify(data,classarra['Thresh'], classarra['Dim'],classarra['Thesin'])
        aggclass += classarra[i]['alpha']*classest
    return np.sign(aggclass)


#######TEST#########
data = np.matrix([[ 1. ,  2.1],
[2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
label =np.matrix( [1.0, 2, -1.0, -1.0, 1.0]).T
bestclass, bestStump, minerror =buildclassify(data,label,D,10)





        
    
        