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
from pathlib import Path

from matplotlib import pyplot as plt

import tools as t
import shape as sh

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

BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / "Rapports" / "Capture_ecran_TP1_RecoFormes"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)



""" 
mySignatureAnalysis 
Input : param_test, liste des paramètres de forme calculés sur les objets à reconnaitre
Recherche les max locaux de la signature 
Analyse des max locaux pour déterminer le type de forme 
"""
def mySignatureAnalysis(param, return_details=False):
    nb_objets=len(param)
    nb_cercles=0
    nb_triangles=0
    nb_octogones=0
    details=[]

    for i in range(nb_objets): 
        # la signature est disponible dans param
        signature=param[i][4]

        if not isinstance(signature, list):
            signature_list = signature.tolist() if hasattr(signature, 'tolist') else list(signature)
        else:
            signature_list = signature

        if len(signature_list) == 0:
            details.append({
                "classe": AUTRE,
                "maxima": np.array([], dtype=int),
                "signature": np.array([], dtype=float),
                "std_rel": 0.0,
            })
            continue

        signature_list=np.roll(signature_list, -signature_list.index(min(signature_list)))
        signature = np.asarray(signature_list, dtype=float)

        # ajout 06/02/25
        # filtre gaussien
        longueur_contour=len(signature)    
        sigma=max(longueur_contour/128, 1e-6)
        signature=sc.ndimage.gaussian_filter(signature, sigma, order=0)
        
        # Trouver les maxima locaux de la signature
        maxima = argrelextrema(signature, np.greater)[0]
        nb_maxima = len(maxima)
        std_rel = float(np.std(signature) / (np.mean(signature) + 1e-8))

        # Cercle : peu de maxima (signature régulière)
        if std_rel < 0.03 and nb_maxima <= 6:
            classe = CERCLE
            nb_cercles += 1
        # Triangle : 3 maxima
        elif nb_maxima == 3:
            classe = TRIANGLE
            nb_triangles += 1
        # Octogone : 8 maxima
        elif 7 <= nb_maxima <= 9:
            classe = OCTOGONE
            nb_octogones += 1
        # Autre
        else:
            classe = AUTRE

        details.append({
            "classe": classe,
            "maxima": maxima,
            "signature": signature,
            "std_rel": std_rel,
        })

    if return_details:
        return(nb_cercles, nb_triangles, nb_octogones, details)
    return(nb_cercles, nb_triangles, nb_octogones)


def save_debug_plot(image_id, im_bgr, mask_bin, imshapes, details):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Analyse panneaux - image {image_id:02d}", fontsize=12)

    axs[0, 0].imshow(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Image source")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask_bin, cmap="gray")
    axs[0, 1].set_title("Binarisation couleur cible")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(imshapes, cmap="gray")
    axs[1, 0].set_title("Objets retenus")
    axs[1, 0].axis("off")

    ax_sig = axs[1, 1]
    if len(details) == 0:
        ax_sig.text(0.5, 0.5, "Aucun objet detecte", ha="center", va="center")
    else:
        class_names = {
            CERCLE: "cercle",
            TRIANGLE: "triangle",
            OCTOGONE: "octogone",
            AUTRE: "autre",
        }
        for idx, d in enumerate(details):
            s = d["signature"]
            if s.size == 0:
                continue
            ax_sig.plot(s, linewidth=1.2, label=f"obj{idx+1} ({class_names[d['classe']]})")
            if len(d["maxima"]) > 0:
                ax_sig.scatter(d["maxima"], s[d["maxima"]], s=12)

    ax_sig.set_title("Signatures lissees + maxima")
    ax_sig.set_xlabel("Index contour")
    ax_sig.set_ylabel("Distance au centre")
    ax_sig.grid(True, alpha=0.3)
    if len(details) > 0:
        ax_sig.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"panneau_debug_im{image_id:02d}.png", dpi=180)
    plt.close(fig)


def save_counts_plot(pred_c, pred_t, pred_o):
    x = np.arange(FIRST_IM, LAST_IM + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(x, gt_cercle, "o-", label="GT cercles")
    axs[0].plot(x, pred_c, "s--", label="Pred cercles")
    axs[0].set_ylabel("Nb")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    axs[1].plot(x, gt_triangle, "o-", label="GT triangles")
    axs[1].plot(x, pred_t, "s--", label="Pred triangles")
    axs[1].set_ylabel("Nb")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    axs[2].plot(x, gt_octogone, "o-", label="GT octogones")
    axs[2].plot(x, pred_o, "s--", label="Pred octogones")
    axs[2].set_xlabel("Indice image")
    axs[2].set_ylabel("Nb")
    axs[2].grid(alpha=0.3)
    axs[2].legend()

    fig.suptitle("Comptage GT vs predictions par image")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "panneaux_comptage_vs_gt.png", dpi=180)
    plt.close(fig)

    


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
AFFICHAGE=False


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
    cluster_idx=np.argmin(dist, axis=0)
    couleur=np.uint8(np.where(np.equal(imlabels, cluster_idx), 255, 0))
    
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
    nb_c,nb_t, nb_o, details = mySignatureAnalysis(param, return_details=True)
    print("cercles : ",  nb_c, "triangles: ", nb_t, "octo: ", nb_o )
    p_cercle.append(nb_c)
    p_triangle.append(nb_t)
    p_octogone.append(nb_o)

    # Sauvegarde de figures de debug exploitables dans le compte rendu.
    if iter in [6, 12, 19]:
        save_debug_plot(iter, im, couleur, imshapes, details)
    
    if(AFFICHAGE):
        #cv2.imshow('Original image', im)     
        #cv2.imshow('Apres conversion', imrgb)
        #cv2.imshow('Classes ', (imclasses))      
        #cv2.imshow('Clustering (labels)', np.uint8(cv2.equalizeHist(imlabels))) 
        #cv2.imshow('Couleur Panneau', couleur)    
        #cv2.normalize(regions, regions_show, 0, 255, cv2.NORM_MINMAX)
        #cv2.imshow('Regions', np.uint8(regions_show*255/K))
        cv2.imshow('Shapes', np.uint8(imshapes))  
        cv2.waitKey(1)

    iter+=1
    

# 7.2 Calcul des "true positive" (i.e., les prédictions correctes) + mesure d'accuracy.
TP=0
print("SIGNATURE :")       
prediction_triplets=list(zip(p_cercle, p_triangle, p_octogone))
ground_truth_triplets=list(zip(gt_cercle, gt_triangle, gt_octogone))

TP=sum(int(p == g) for p, g in zip(prediction_triplets, ground_truth_triplets))
accuracy=100 * TP / len(ground_truth_triplets)

print("prediction :\t", prediction_triplets)
print("ground truth :\t", ground_truth_triplets)
print("Accuracy: {:.1f}%".format(accuracy))

save_counts_plot(p_cercle, p_triangle, p_octogone)
    

cv2.destroyAllWindows()
