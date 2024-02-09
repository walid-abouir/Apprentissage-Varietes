# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:58:58 2023

@author: aboui
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

def oracle1(x):
    f= 2*((x[0]+x[1]-2)**2)+(x[0]-x[1])**2
    df=np.zeros(2)
    df[0]=6*x[0]+2*x[1]-8
    df[1]=6*x[1]+x[0]-8
    Hf=np.zeros( (2,2) )
    Hf[0,0]=6
    Hf[0,1]=2
    Hf[1,0]=2
    Hf[1,1]=6
    
    return f,df,Hf

def Gradient(function,h=1e-1,xini=np.array([0,0]),eps=1.e-6):
    iter=0
    x=np.copy(xini)
    xiter=[x]
    itermax=10000
    err=2*eps
    #xiter=[function(xini)[0]]
    while err>eps and iter<itermax:
        f,df,Hf=function(x)
        x=x-h*df
        xiter.append(np.copy(x))
        err=np.linalg.norm(df) #norme du gradient de f
        iter=iter+1
        print(iter,err)
    y=np.array(xiter)
    return x,y,iter

