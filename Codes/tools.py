#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:01:45 2022

@author: gouiffes
"""
# librairies
import cv2
import numpy as np

def myResize(frame, scale_percent):
    my_h, my_w = frame.shape[0], frame.shape[1]
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    im = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return(im)



""" Conversion espace couleur normalise rgb
    im : image codee en BGR
    r=R/(R+G+B)
    g=G/(R+G+B)
    b=B/(R+G+B)
     """
def myConvert(im):
    B= np.float32(im[:,:,0])
    G= np.float32(im[:,:,1])
    R= np.float32(im[:,:,2])
    den = np.float32(R+G+B+0.0001)
    imrgb=np.float32(im)

    imrgb[:,:,0]=cv2.divide(B, den)
    imrgb[:,:,1]=cv2.divide(G, den)
    imrgb[:,:,2]=cv2.divide(R, den)
    return(imrgb)



""" Mise en forme des donnees pour appliquer le kmeans
 renvoie :
     - une image de labels de classes 0, 1, 2...K-1
     - une images de K couleurs (1 couleur = 1 centre de classe)
     """
def myKmeans(im, k):
    couleurs = im.reshape((-1, 3))
    couleurs = np.float32(couleurs)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, (centers) = cv2.kmeans(couleurs, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # imlabels est une image de la meme taille que im mais avec 1 seul canal 
    imlabels=np.zeros( ( im.shape[0], im.shape[1]),  np.uint8)
    imlabels =np.uint8( labels.reshape(imlabels.shape))
    
    # convert all pixels to the color of the centroids
    imclasses = (centers[labels.flatten()])
    # reshape back to the original image dimension
    imclasses = imclasses.reshape(im.shape)
    #print(imclasses.type)
    return(imlabels, imclasses, centers)
