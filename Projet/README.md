# Compte rendu - Reconnaissance de panneaux (GTSRB)

## Contexte et objectif
Ce projet vise a reconnaitre automatiquement des panneaux de signalisation a partir d'images. L'objectif principal est d'atteindre une bonne accuracy (taux de bonnes predictions) sur le jeu de donnees GTSRB.

## Jeu de donnees
- Source : GTSRB (German Traffic Sign Recognition Benchmark)
- Nombre d'images utilisees (train) : 26640
- Nombre de classes : 43

Remarque : un mode fallback existe avec un petit jeu local et des pseudo-labels (etiquettes estimees automatiquement), mais ce mode est trop bruite pour viser 90% d'accuracy.

## Methodes
1) Approche classique (HOG + SVM)
- HOG : descripteur de forme (resume les contours dans l'image).
- SVM : classifieur supervise qui separe les classes.

2) CNN maison
- Reseau de neurones convolutionnel avec augmentation de donnees.
- Ajouts : BatchNorm (stabilise l'entrainement) et Dropout (limite le surapprentissage).

3) Transfer learning (MobileNetV2)
- Reseau pre-entraine sur ImageNet, adapte a GTSRB.

## Resultats principaux (GTSRB complet)
- CNN maison (avant fine-tuning) : ~0.9146 d'accuracy test.
- CNN maison apres fine-tuning : ~0.9217 d'accuracy test.

Le fine-tuning consiste a continuer l'entrainement avec un taux d'apprentissage plus faible pour affiner les poids.

## Analyse rapide
- Le CNN apprend bien sur GTSRB car le dataset est grand et bien labelise.
- Les pseudo-labels locaux degradent fortement les performances.
- Le fine-tuning apporte un gain net pour depasser 90%.

## Instructions d'execution
1) Ouvrir le notebook : Projet_panneaux.ipynb
2) Lancer les cellules dans l'ordre
3) Verifier la cellule finale d'evaluation pour l'accuracy test

## Limites
- Temps d'entrainement assez long sur CPU.
- Les performances chutent si on n'utilise pas le GTSRB complet.