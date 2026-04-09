#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:45:08 2022

@author: gouiffes
"""
import cv2
import numpy as np

def display(titre, im):
    m = np.asanyarray(im, np.uint8)
    #cv2.Convert(im, m)
    cv2.imshow(titre, m)


#TODO : exercice 1
""" -------------------------------------------------------
Simple Background Subraction  :
- Bt : background image
- It : current image
- thres = seuil
- filter = 0(resp. 1) without (resp with) morphological filtering
---------------------------------------------------------
"""
def background_subtract(background, image, thres):
    # convertir image / background (fond) en np.asarray (np.float32)
    image = np.asarray(image, np.float32)
    background = np.asarray(background, np.float32)

    diff = np.abs(image - background)
    b = np.where(diff > float(thres), 255, 0).astype(np.uint8)

    return(b)


#TODO : exercice 2
""" -------------------------------------------------------
Normalisation photométrique
- Bt : background image
- It : current image
- filter = 0(resp. 1) without (resp with) morphological filtering
---------------------------------------------------------
"""
def photometric_normalization(image):
    im = np.asarray(image, np.float32)
    mu = float(np.mean(im))
    sigma = float(np.std(im))
    sigma = max(sigma, 1e-6)

    # Normalisation puis remise dans une dynamique 8 bits exploitable.
    z = (im - mu) / sigma
    z = cv2.normalize(z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_image = z.astype(np.uint8)
    return(norm_image)



#TODO : exercice 3
""" -------------------------------------------------------
Background update using a temporal mean
- background : image of the current background
- image : current image of the video
---------------------------------------------------------"""
def update_beackground_mean(background, image, alpha):
    
    image = np.asarray(image, np.float32)
    background = np.asarray(background, np.float32)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    # Bt = alpha * B(t-1) + (1-alpha) * It
    res = alpha * background + (1.0 - alpha) * image
    res = np.clip(res, 0, 255).astype(np.uint8)
    return(res)



""" -------------------------------------------------------
Background update using a median value of N images
- background : image of the current background
- im_list : list image
---------------------------------------------------------"""
def update_beackground_median(im_list):
    n= len(im_list)
    print(n)
    if (n>0):
        image=im_list[0]
        h=image.shape[0]
        w=image.shape[1]
        res=np.zeros((h, w, 3), np.float32) 
        val=np.zeros((n, 3), np.float32) 

        for i in range(h): 
            for j in range(w):
                for k in range(n):
                    val[k]=im_list[k][i,j]
                val[:,0] =np.sort(val[:,0])
                val[:,1] =np.sort(val[:,1])
                val[:,2] =np.sort(val[:,2])
                res[i,j]=val[int(n/2)] 
    return(res)







""" -------------------------------------------------------
Detection d'objets passant dans une zone rectangulaire
- objets : images de composantes connexes (objets en mouvement)
- region  : rectangle ROI
---------------------------------------------------------"""
def objects_entering_exiting(objects, r):
    # si objet est a l fois dans la region et touche le bord : sortant
    # si objet est àla fois hors de la region et touche le bord : entrant
    nb_out=0
    nb_in =0
    # imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # area= 
    # image = cv2.rectangle(image, start_point, end_point, color, thickness)
    # bord = 
    return(nb_in, nb_out)