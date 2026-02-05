#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:01:45 2022

@author: gouiffes
"""
# librairies

import cv2
import numpy as np
import scipy as sc
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import tools as t



AFFICHAGE = True

""" Signature de la forme 
Input : objet, image binaire contenant une composante connexe
"""
def mySignature(objet):
    contour,_ = cv2.findContours(objet,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt=contour[0]
    ch=np.mean(cnt[:, 0, 0])
    cw=np.mean(cnt[:, 0, 1])
    signature=[]
   
    # Calcul de la distance de chaque point du contour au centre de gravité
    for point in cnt:
        x = point[0, 0]
        y = point[0, 1]
        distance = np.sqrt((x - ch)**2 + (y - cw)**2)
        signature.append(distance)
    
    return(signature)


"""Codage de Freeman 
Input : objet, image binaire contenant une composante connexe
"""
Freeman =np.array( [[3, 2, 1], [4, -1, 0], [5, 6, 7]] )

def myFreemanCode(objet):
    contour,_ = cv2.findContours(objet,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt=contour[0]

    p0 = cnt[0]
    code=[]
    for p in range(1, len(cnt)):
        p1 = cnt[p]
        # vecteur d=[dx,dy]=[x−xprev​,y−yprev​]
        dx = p1[0, 0] - p0[0, 0]
        dy = p1[0, 1] - p0[0, 1]
        # Normaliser les valeurs à -1, 0, ou 1
        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        code.append(Freeman[dy+1, dx+1])
        p0 = p1
    
    return(code)


"""Perimetre 
Input : code de freeman du contour 
"""
def myPerimeter(freeman_code):
    perim=0.
    for p in range(0, len(freeman_code)):
        # Les directions paires (0,2,4,6) ont une distance de 1
        # Les directions impaires (1,3,5,7) ont une distance de sqrt(2)
        if freeman_code[p] % 2 == 0:
            perim += 1.0
        else:
            perim += np.sqrt(2)

    return(perim)

"""mySkeleton
Input : binary image to be skeletonized
"""
def mySkeleton(im):
    # Step 1: Create an empty skeleton
    size = np.size(im)
    skel = np.zeros(im.shape, np.uint8)
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    while True:
        O = cv2.morphologyEx(im, cv2.MORPH_OPEN, B)
        temp = cv2.subtract(im, O)
        eroded = cv2.erode(im, B)   
        skel = cv2.bitwise_or(skel,temp)
        im = eroded.copy()
        if cv2.countNonZero(im)==0:
            break
    # Displaying the final skeleton
    cv2.imshow("Skeleton",skel)
    cv2.waitKey(0)
    

""" Calcul des parametres de forme
En entree : une image binaire 1 : objet, 0 fond
En sortie : une liste de parametres
     """
def myShapeCompute(objet, stats):
    S=[] # parametres de forme
    aire = stats[4]  # L'aire est disponible dans stats[4] (cv2.CC_STAT_AREA)
    print("AIRE :", aire)
    pseudorect = aire/ ( stats[2]* stats[3])
    moments = cv2.moments(objet) 
    hu_moments = cv2.HuMoments(moments)
    #for i in range(0, 7):
      #  hu_moments[i] = -1* np.sign(hu_moments[i]) * np.log(abs(hu_moments[i]))
    print("MOMENTS")
    print("mu20 :", moments['mu20'])
    print("mu11 :", moments['mu11'])
    print("mu02 :", moments['mu02'])
    print("mu03 :", moments['mu03'])
    print("mu21 :", moments['mu21'])
    print("mu12 :", moments['mu12'])    
    print("mu03 :", moments['mu03'])
    print("nu20 :", moments['nu20'])
    print("nu11 :", moments['nu11'])
    print("nu02 :", moments['nu02'])    
    print("nu30 :", moments['nu30'])
    print("nu21 :", moments['nu21'])
    print("nu12 :", moments['nu12'])    
    print("nu03 :", moments['nu03'])
    
    freeman_code=myFreemanCode(objet)
    perimetre= myPerimeter(freeman_code)
    compacite = 4*np.pi*aire/(perimetre**2)
    sig=mySignature(objet)

    S.append(aire)
    S.append(perimetre)
    S.append(compacite)
    S.append(pseudorect)
    S.append(sig)
    S.append(moments)
    S.append(hu_moments)    
    
  
    return(S)
