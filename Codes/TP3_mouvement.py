#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2 16:42:36 2022

@author: gouiffes
"""

# librairies
import cv2
import numpy as np
from matplotlib import pyplot as plt
import motion

"""
Choisir une vidéo 
"""
#cap = cv2.VideoCapture('im/rouen_video.avi')
#cap = cv2.VideoCapture('im/atrium_video.avi')
#cap = cv2.VideoCapture('im/sherbrooke_video.avi')
cap = cv2.VideoCapture('../images/videos/campus.mp4') # MODIFIER LE CHEMIN ICI SI NÉCESSAIRE
#cap = cv2.VideoCapture('im/lego.mp4')

# ouverture du flux video
ret, frame = cap.read()
# pour l'affichage du flot optique
hsv = np.zeros_like(frame)
hsv[..., 1] = 255

# conversion en niveaux de gris
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

#TODO : ex2 : coder cette fonction dans motion.py
# pour réduire les changements d'éclairage (l'égalisation fonctionne aussi)
# frame=motion.photometric_normalization(frame)


my_h, my_w = frame.shape[0], frame.shape[1]
scale_percent =100
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
im = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


grey_image= np.asanyarray(im, np.float32)
background = np.asanyarray(im, np.uint8)
foreground = np.asanyarray(im, np.uint8)
regions = np.zeros(dim, np.uint16)


im_list=[]
k=0
ret=True
# boucle de lecture frame par frame
while (ret==True):
	# stocker l'image issue de la vidéo à l'instant t dans la variable "frame"
    ret, frame = cap.read() 
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    
    # frame=motion.photometric_normalization(frame) (ex 2)

    if(ret==True):
        # redimensionnement image     
        im = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        """ 1- Detection des pixels en mouvement """ 
        # TODO : ex1
        foreground=motion.background_subtract(background, im,20)
        """"2- mise a jour du fond """
        # TODO : ex3
        background=motion.update_beackground_mean(background, im, 0.8)
        cv2.imshow('fond', np.uint8(background))

        """3 morphologie mathematique"""
        #TODO: ex5
        # A COMPLETER (cf fonction cv2.morphologyEx())
        # Définition du kernel utilisé pour les opérations
        kernel = np.ones((15,15),np.uint8)

        """ 4 etiquetage en composantes connexes """
        im2 = np.uint8(foreground)    #TODO : ici ex 5, modifier la variable 'foreground' avec le résultat à l'étape 3 précédente
        retcc, regions, stats, centroids = cv2.connectedComponentsWithStats(im2)
        print("Nombre d'objets", len(stats)-1)
        
        """ 5 Flot Optique """
        #TODO: ex6
        # A COMPLETER FLOT OPTIQUE

        # INFORMATION 
        # print(k, " stats ", stats)  # Affiche les statistiques pour chaque étiquette (y compris l’étiquette du fond)
        # stats(label, COLONNE) où COLONNE correspond à : 
        # 0 : coordonnée x du coin supérieur gauche
        # 1 : coordonnée y du coin supérieur gauche
        # 2 : hauteur de la zone
        # 3 : largeur de la zone
        # 4 : surface (nombre de pixels)
        # print(stats)  # Affiche la matrice complète des statistiques
                
        """ 5 affichage """
        cv2.imshow('image', np.uint8(frame))
        cv2.waitKey(0)
       
        # quiter la boucle infinie lorqu'on appuie sur la touche 'q'
        if(cv2.waitKey(40) & 0xFF == ord('q')):
            break
        k+=1
 
#quiter le programme et fermer toutes les fenêtres ouvertes
cap.release()
cv2.destroyAllWindows()
