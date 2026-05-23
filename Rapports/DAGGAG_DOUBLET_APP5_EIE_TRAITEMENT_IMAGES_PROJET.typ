#set page(margin: (x: 2.2cm, y: 2.2cm))
#set par(justify: true)
#set text(lang: "fr")

= Projet : Reconnaissance de panneaux de signalisation (GTSRB)

*Date :* 23 Mai 2026\
*Anas DAGGAG - Jeremy DOUBLET - APP5 EIE*

#image("/assets/image-1.png", width: 20%)
== Objectif

Ce projet vise a reconnaitre automatiquement des panneaux de signalisation a partir d'images. L'objectif principal est d'obtenir une bonne precision de classification sur le jeu de donnees GTSRB, et de comparer plusieurs approches :
- methodes classiques de reconnaissance de formes,
- modele supervise (CNN maison),
- modele pre-entraine (transfer learning).

== Problematique et demarche

Problematique : comment reconnaitre des panneaux dans des images reelles qui presentent des variations de luminosite, de point de vue, d'occlusion et de bruit ?

== Partie 1 : Presentation du jeu de donnees

=== 1) Donnees

Le dataset GTSRB contient des images de panneaux reparties sur 43 classes. Dans notre execution, le chargement donne :
- 26 640 images d'entrainement,
- 12 630 images de test,
- 43 classes.

Les images ont des tailles variables. On applique un redimensionnement en $32 times 32$ pour l'apprentissage.

=== 2) Split du jeu de donnees

Pour l'entrainement du CNN, on utilise :
- un split interne $90/10$ (train/validation) via `validation_split=0.1`.

=== 3) Qualite et pretraitements

Voici les différents point relevée concernant la qualite du dataset :
- dataset reel, avec variations d'eclairage et de cadrage,
- classes desequilibrees,
- presence de bruit et d'arrieres-plans variables
- images pixellisees.

#figure(
	image("Capture_ecran_Projet/pixelisation.png", width: 50%),
	caption: [Exemple d'image pixelisee dans GTSRB.]
)

Interpretation : on observe des contours en "marches d'escalier". Cela signifie que l'image est composee de gros pixels visibles, ce qui rend les formes moins nettes et peut rendre la classification plus difficile.

== Partie 2 : Methodes classiques de reconnaissance de formes

On teste une approche par segmentation de couleur (HSV) puis analyse des contours :
- masques pour les couleurs principales (rouge, jaune, bleu),
- morphologie mathematique pour nettoyer le masque,
- extraction de contour principal,
- signature de contour pour estimer la forme (cercle, triangle, octogone, autre).

Limites :
- sensible aux variations de couleur et d'eclairage,
- ne gere pas bien les panneaux complexes,
- methode utile pour une pre-classe grossiere mais pas suffisante pour 43 classes.

#figure(
	image("Capture_ecran_Projet/Approche classique - resultat test.png", width: 60%),
	caption: [Resultats de l'approche classique (descripteurs + SVM).]
)

Interpretation : l'approche classique atteint une accuracy elevee (taux de bonnes predictions) et des scores precision/recall proches. Cela indique que, sur ce jeu local, les predictions sont globalement fiables et equilibrees.

== Partie 3 : Methode supervisee (CNN maison)

=== 1) Architecture

CNN simple avec :
- 3 blocs Conv2D + BatchNorm + ReLU + MaxPooling,
- Flatten,
- Dense(256) + Dropout,
- sortie softmax sur 43 classes.

=== 2) Reglages

- Optimizer : Adam
- Learning rate initial : $1e-3$
- Batch : 64
- Epoques : 15 + fine-tuning 8 epoques
- EarlyStopping + ReduceLROnPlateau

=== 3) Resultats

Sur GTSRB complet :
- accuracy (taux de bonnes predictions) test avant fine-tuning : 0.9146
- accuracy (taux de bonnes predictions) test apres fine-tuning : 0.9217

Le fine-tuning (learning rate plus faible) permet d'ameliorer legerement le score au-dela de 90%.

==== a) CNN sur GTSRB (captures)

#figure(
	image("Capture_ecran_Projet/accuracy_cnn_gtsrb.png", width: 85%),
	caption: [Courbe d'accuracy du CNN sur GTSRB (entrainement vs validation).]
)

Interpretation : la courbe de validation (donnees gardees pour evaluer pendant l'entrainement) monte vite et reste proche de la courbe d'entrainement. L'ecart reste limite, ce qui suggere peu de surapprentissage (overfitting : le modele memorise trop et generalise mal).

#figure(
	image("Capture_ecran_Projet/predictions_cnn_gtsrb.png", width: 50%),
	caption: [Exemples de predictions du CNN sur GTSRB.]
)

Interpretation : la plupart des panneaux sont correctement classes. Quelques erreurs restent possibles sur des images floues ou peu contrastees, ce qui est attendu dans un jeu reel.

==== b) CNN maison sur dataset local

#figure(
	image("Capture_ecran_Projet/courbes_cnn_maison.png", width: 85%),
	caption: [Courbes loss/accuracy du CNN maison (dataset local).]
)

Interpretation : l'accuracy reste faible et progresse peu. Cela montre que le CNN maison n'arrive pas a apprendre des caracteristiques robustes avec ce petit jeu.

#figure(
	image("Capture_ecran_Projet/confusion_cnn_maison.png", width: 55%),
	caption: [Matrice de confusion du CNN maison (dataset local).]
)

Interpretation : la diagonale (bonnes predictions) est faible et beaucoup de valeurs sont hors diagonale. Cela signifie que le modele confond souvent les classes.

== Partie 4 : Modele pre-entraine (transfer learning)

Un modele MobileNetV2 pre-entraine (ImageNet) est adapte au dataset.
Principe :
- on geler les premieres couches,
- on entraine une tete de classification,
- on degeler partiellement pour affiner.

Cette approche est interessante car elle permet d'obtenir de bonnes performances avec moins de donnees, mais elle demande un ajustement fin des hyperparametres et du nombre de couches degelées.

Resultats (test) :
- accuracy (taux de bonnes predictions) : 0.8859
- precision (fiabilite des predictions positives) : 0.8905
- recall (taux de panneaux correctement retrouves) : 0.8859
- f1 (moyenne entre precision et recall) : 0.8863

#figure(
	image("Capture_ecran_Projet/f1_comparison.png", width: 80%),
	caption: [Comparaison des scores F1 sur le dataset local.]
)

Interpretation : le F1 (moyenne entre precision et recall) est plus eleve pour l'approche classique que pour le CNN maison, et le transfer learning se situe entre les deux. Cela indique que, sur peu de donnees, les methodes classiques restent tres competitives.

== Partie 5 : Conclusion

- Les methodes classiques de formes sont utiles pour une detection grossiere, mais insuffisantes pour une classification fine.
- Le CNN maison atteint 0.9217 d'accuracy (taux de bonnes predictions) sur GTSRB apres fine-tuning.
- Le transfer learning est une alternative efficace, a condition de bien regler le fine-tuning.

Perspectives :
- augmenter les augmentations de donnees,
- tester un backbone plus puissant (ResNet, EfficientNet),
- sauvegarder les modeles (format .pth si implementation PyTorch, sinon .keras/.h5 pour TensorFlow).

== Annexe : fichiers principaux

```txt
Projet/Projet_panneaux.ipynb
Rapports/DAGGAG_DOUBLET_APP5_EIE_TRAITEMENT_IMAGES_PROJET.typ
```