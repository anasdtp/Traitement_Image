#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:42:36 2022

@author: gouiffes
"""

# librairies

import cv2
import numpy as np
import scipy as sc
from scipy.signal import argrelextrema

from matplotlib import pyplot as plt

import tools as t
import shape as sh
import TP1_recoFormes as recoFormes

CERCLE=0
TRIANGLE=1
OCTOGONE=2
AUTRE=3
NB_EXAMPLES=0
formes=["cercle", "triangle", "octogone", "autre"]

# Vérité terrain / Ground Truth
gt_cercle=[0,0, 0, 1, 0, 0, 3, 0, 0, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0]
gt_triangle=[0,0, 2, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1 ]
gt_octogone=[1,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1,0]

# Liste à compléter avec vos prédictions
p_cercle=[] 
p_triangle=[]
p_octogone=[]



""" 
mySignatureAnalysis 
Input : param_test, liste des paramètres de forme calculés sur les objets à reconnaitre
Recherche les max locaux de la signature 
Analyse des max locaux pour déterminer le type de forme 
"""
def mySignatureAnalysis(param):
    nb_objets=len(param)
    prediction=np.zeros(shape=(nb_objets), dtype=np.uint8)
    nb_cercles=0
    nb_triangles=0
    nb_octogones=0
    nb_autres=0
    TP=0
    for i in range(nb_objets): #(5, LAST_IM+1):
        pass 
        #TODO

    for i in range(nb_objets): 
        # la signature est disponible dans param_test
        signature=param[i][4]

        if not isinstance(signature, list):
            signature_list = signature.tolist() if hasattr(signature, 'tolist') else list(signature)
        else:
            signature_list = signature

        signature_list=np.roll(signature_list, -signature_list.index(min(signature_list)))
        signature = np.asarray(signature_list)

        # ajout 06/02/25
        # filtre gaussien
        longueur_contour=len(signature)    
        sigma=longueur_contour/128
        signature=sc.ndimage.gaussian_filter(signature, sigma, order=0)
        
        #plt.plot(signature)
        #plt.ylabel('some numbers')
        #plt.show()  
        
        # Trouver les maxima locaux de la signature
        maxima = argrelextrema(signature, np.greater)[0]
        nb_maxima = len(maxima)
        #print("nb maxima : ", maxima)
        # Cercle : peu de maxima (signature régulière)
        if nb_maxima <= 2:
            nb_cercles += 1
        # Triangle : 3 maxima
        elif nb_maxima == 3:
            nb_triangles += 1
        # Octogone : 8 maxima
        elif nb_maxima >= 7 and nb_maxima <= 9:
            nb_octogones += 1
        # Autre
        else:
            prediction[i] = AUTRE



    # A COMPLETER : recuperez votre fonction signature et adaptez la

    return(nb_cercles, nb_triangles, nb_octogones)

    


""" 
Paramètres image 
"""
path = "im/panneaux/"
FIRST_IM=1
LAST_IM=19

# Expliquer le rôle de cette variable
K=3 

# Expliquer le rôle de cette variable
c=[0, 0, 1]

# taille min des objets
TAILLE_MIN=200
# AFFICHAGE
AFFICHAGE=True


iter=FIRST_IM
while(iter<=LAST_IM):
    # 1. Ouverture et affichage image 
    path_im= path+ "im"+  f"{iter:0>2d}.png"
    frame = cv2.imread(path_im, cv2.IMREAD_COLOR)
    im=t.myResize(frame, 200)

    # 2. Expliquer cette étape
    #convertie l'image en couleur RGB normalisée 
    imrgb=t.myConvert(im)
   
    #3. Expliquer cette étape
    #applique la méthode des k-means pour segmenter l'image en K classes de couleurs
    imlabels, imclasses, centers = t.myKmeans(imrgb, K)
  
    # 4. Expliquer cette étape
    dist = np.sum(abs(centers - c), axis=1)
    k=np.argmin(dist, axis=0)   
    couleur=np.uint8(np.where(np.equal(imlabels, k), 255, 0))
    
    #5. Expliquer cette étape
    contour,_ = cv2.findContours(couleur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    imshapes=couleur
    for cnt in contour:
        cv2.drawContours(imshapes,[cnt],0,255,-1)  

    retcc, regions, stats, centroids = cv2.connectedComponentsWithStats(imshapes)
    nb_regions=len(stats)
    regions_show= np.asanyarray(regions, np.uint8)
  
    # 6. Faire l'analyse
    param=[]
    for r in range(1, nb_regions):
        # Expliquer cette étape
        if(stats[r,4] >TAILLE_MIN):     
            objet=np.uint8(np.where(np.equal(regions, r), 255, 0))
            start_point=(stats[r,0], stats[r,1])
            end_point=(stats[r,0]+ stats[r, 2], stats[r,1]+ stats[r,3])
            color = (127, 0, 0)
            S=sh.myShapeCompute(objet, stats[r, :])
            param.append(S)

    # 7.1 Commenter vos résultats
    nb_c,nb_t, nb_o= mySignatureAnalysis(param)
    print("cercles : ",  nb_c, "triangles: ", nb_t, "octo: ", nb_o )
    p_cercle.append(nb_c)
    p_triangle.append(nb_t)
    p_octogone.append(nb_o)
    
    if(AFFICHAGE):
        #cv2.imshow('Original image', im)     
        #cv2.imshow('Apres conversion', imrgb)
        #cv2.imshow('Classes ', (imclasses))      
        #cv2.imshow('Clustering (labels)', np.uint8(cv2.equalizeHist(imlabels))) 
        #cv2.imshow('Couleur Panneau', couleur)    
        #cv2.normalize(regions, regions_show, 0, 255, cv2.NORM_MINMAX)
        #cv2.imshow('Regions', np.uint8(regions_show*255/K))
        cv2.imshow('Shapes', np.uint8(imshapes))  
    cv2.waitKey(k)

    iter+=1
    

# 7.2 Calcul des "true positive" (i.e., les prédictions correctes) + mesure d'accuracy.
TP=0
print("SIGNATURE :")       
print("prediction :\t",)
print("ground truth :\t", ) 
print("Accuracy:",   ) 
    

cv2.destroyAllWindows()
