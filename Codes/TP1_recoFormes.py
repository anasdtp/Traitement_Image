#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:42:36 2022

@author: gouiffes
"""

# bibliotheque 
import cv2
import numpy as np
import scipy as sc
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

#fichiers "maison"
import tools as t
import shape as sh

CERCLE=0
TRIANGLE=1
OCTOGONE=2
CARRE=3
RECTANGLE =4
AUTRE=5

formes=["cercle", "triangle", "octogone", "carre", "rectangle", "autre"]
ground_truth=np.array([2,  2, 2, 4, 4, 4, 4, 4, 0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 5, 0, 0])

""" 
Paramètres image 
"""
path = "im/formes/"
FIRST_IM=0
LAST_IM=26
NB_EXAMPLES=5

# taille min des objets
TAILLE_MIN=50
# AFFICHAGE
AFFICHAGE=True

""" dist=
myCompacityAnalysis 
- param_test  : liste des paramètres de forme calculés sur les objets à reconnaitre
- param_ex : liste des paramètres de forme calculés sur les exemples 
"""
def myCompacityAnalysis(param_test, param_ex):
    nb_objets=len(param_test) # à comparer avec les exemples
    nb_ex=len(param_ex)
    prediction=np.zeros(shape=(nb_objets), dtype=np.uint8) # tableau vide à remplir au fur et à mesure avec les prédictions pour chaque image
    TP=0 # vérité terrain
    compacite=[] #garder en mémoire le résultat de compacité de chaque image

    # Pour chaque objet à tester
    for i in range(nb_objets):
        comp_test = param_test[i][2]  # Compacité de l'objet à tester (indice 2)
        compacite.append(round(comp_test, 2))
        
        # Trouver l'exemple le plus proche en termes de compacité
        min_dist = float('inf')
        best_match = 0
        
        for j in range(nb_ex):
            comp_ex = param_ex[j][2]  # Compacité de l'exemple
            dist = abs(comp_test - comp_ex)
            
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        prediction[i] = best_match
        
        # Vérifier si la prédiction est correcte
        if prediction[i] == ground_truth[i]:
            TP += 1

   
    print("COMPACITE :")
    print("prediction \t:\t", prediction)
    print("ground truth :\t", ground_truth) 
    print("compacite: \t", compacite)
    print("Accuracy :", TP/(len(prediction)))
    

""" 
myHuMomentAnalysis 
- param_test  : liste des paramètres de forme calculés sur les objets à reconnaitre
- param_ex : liste des paramètres de forme calculés sur les exemples 
"""
def myHuMomentsAnalysis(param_test, param_ex):
    nb_objets=len(param_test)
    nb_ex=len(param_ex)
    prediction=np.zeros(shape=(nb_objets), dtype=np.uint8)
    TP=0
    
    # Pour chaque objet à tester
    for i in range(nb_objets):
        hu_test = param_test[i][6]  # Moments de Hu de l'objet à tester (indice 6)
        
        # Trouver l'exemple le plus proche en termes de moments de Hu
        min_dist = float('inf')
        best_match = 0
        
        dists = []
        for j in range(nb_ex):
            hu_ex = param_ex[j][6]  # Moments de Hu de l'exemple
            # Calculer la distance euclidienne entre les vecteurs de moments de Hu
            dists.append(float(np.linalg.norm(hu_test - hu_ex)))
            # print(f"Distance entre l'objet {i} et l'exemple {j} : {dist}")
            
        best_match = np.argmin(dists)
        # min_dist = dists[best_match]
        
        prediction[i] = best_match
        
        # Vérifier si la prédiction est correcte
        if prediction[i] == ground_truth[i]:
            TP += 1


    print("MOMENT DE HU :")       
    print("prediction \t:\t", prediction)
    print("ground truth :\t", ground_truth) 
    print("Accuracy:",   TP/(len(prediction)))
    

""" 
mySignatureAnalysis 
Input : param_test, liste des paramètres de forme calculés sur les objets à reconnaitre
Recherche les max locaux de la signature 
Analyse des max locaux pour déterminer le type de forme 
"""
def mySignatureAnalysis(param_test):
    nb_objets=len(param_test)
    prediction=np.zeros(shape=(nb_objets), dtype=np.uint8)# tableau vide à remplir au fur et à mesure avec les prédictions pour chaque image
    TP=0
    for i in range(nb_objets): 
        # la signature est disponible dans param_test
        signature=param_test[i][4]

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
        
        # Trouver les maxima locaux de la signature
        maxima = argrelextrema(signature, np.greater)[0]
        nb_maxima = len(maxima)
        
        # Cercle : peu de maxima (signature régulière)
        if nb_maxima <= 2:
            prediction[i] = CERCLE
        # Triangle : 3 maxima
        elif nb_maxima == 3:
            prediction[i] = TRIANGLE
        # Carré : 4 maxima
        elif nb_maxima == 4:
            # Vérifier si c'est un carré ou un rectangle
            # En analysant la variance des distances entre maxima
            if nb_maxima == 4:
                distances = []
                for k in range(nb_maxima):
                    dist = signature[maxima[k]]
                    distances.append(dist)
                variance = np.var(distances)
                # Si variance faible = carré, sinon rectangle
                print(f"Image {i} - Variance : {variance}, Distance moy : {np.mean(distances)}")
                if variance < (np.mean(distances)/15000.0):
                    prediction[i] = CARRE
                else:
                    prediction[i] = RECTANGLE
        # Octogone : 8 maxima
        elif nb_maxima >= 7 and nb_maxima <= 9:
            prediction[i] = OCTOGONE
        # Autre
        else:
            prediction[i] = AUTRE

        if(prediction[i]==ground_truth[i]):
            TP+=1
    
    print("SIGNATURE :")       
    print("prediction \t:\t", prediction)
    print("ground truth :\t", ground_truth) 
    print("Accuracy:",   TP/(len(prediction)))

"""
myShapeAnalysis
Recupère les paramètres de forme de l'ensemble des images et les analyse 
"""
def myShapeAnalysis(param_test, param_ex):  
    # myCompacityAnalysis(param_test, param_ex)
    # myHuMomentsAnalysis(param_test, param_ex)
    mySignatureAnalysis(param_test)

if __name__ == "__main__":
    param_test=[]
    param_exemples=[]
    scale_percent=100
    """ calcul des parametres de forme """
    iter=FIRST_IM

    while(iter<=LAST_IM):
        # 1. Ouverture et affichage image 
        path_im= path + f"{iter:0>2d}.png"
        frame = cv2.imread(path_im, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Erreur lors de la lecture de l'image {path_im}")
            iter+=1
            continue
        im=t.myResize(frame, scale_percent)
        
        #2.  Binarisation
        ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        contour,_ = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        retcc, regions, stats, centroids = cv2.connectedComponentsWithStats(im) 
        nb_regions=len(stats)
    
        # 3. Calcul de tous les paramètres de forme 
        # les 5 exemples sont stockés dans param_exemples et les images à tester/identifier sont dans param_test
        
        for r in range(1, nb_regions):
            #  Filtrage sur la taille 
            if(stats[r,4] >TAILLE_MIN):
                objet=np.uint8(np.where(np.equal(regions, r), 255, 0))
                # print(f"Image {iter:0>2d} - region {r} : taille = {stats[r,4]}")
                # print(objet)
                S=sh.myShapeCompute(objet, stats[r, :])
                if(iter<NB_EXAMPLES):
                    param_exemples.append(S)
                else:
                    param_test.append(S)
        if(AFFICHAGE):
            cv2.imshow('Original image', im)     
            cv2.waitKey(1)

        iter+=1

    # 4. Analyse des parametres de forme
    myShapeAnalysis(param_test, param_exemples)  

    cv2.destroyAllWindows()