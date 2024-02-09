#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:13:39 2023

@author: munier
"""

import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_name):
    PATH_NAME = os.getcwd()
    return iio.imread(PATH_NAME + "/" + file_name)

PSF = load_image("PSFs.tif")


(K,N,_) = PSF.shape

print("Le tenseur PSF est de taille",PSF.shape)
print("Il y a "+str(K)+" images de taille "+str(N)+" x "+str(N))

#Rectification
PSF -= np.min(PSF,axis = (1,2,)).reshape((PSF.shape[0],1,1,))
PSF = PSF / np.max(PSF,axis = (1,2,)).reshape((PSF.shape[0],1,1,)) # / np.linalg.norm(PSF,axis = (1,2,)).reshape((PSF.shape[0],1,1,))

PSF_min, PSF_max = PSF.min(), PSF.max()
n1, n2 = 10,15
plt.figure()
for k in range(n1*n2):
    
    plt.subplot(n1,n2,k+1)
    plt.imshow(PSF[k,:,:], vmin=PSF_min, vmax=PSF_max)
    plt.axis('off')

plt.show()









