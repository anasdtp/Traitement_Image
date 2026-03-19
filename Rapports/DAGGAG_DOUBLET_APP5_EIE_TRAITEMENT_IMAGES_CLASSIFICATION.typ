#set page(margin: (x: 2.2cm, y: 2.2cm))
#set par(justify: true)
#set text(lang: "fr")

= Rapport TP2 : Classification

*Date :* 19 Mars 2026\
*Anas DAGGAG - Jeremy DOUBLET - APP5 EIE*

== Objectif

Ce TP a pour but de comparer plusieurs approches de classification d'images sur Fashion-MNIST :
- un MLP de base,
- un MLP avec hyperparametres modifies,
- un CNN simple.

Le notebook fourni a ete conserve, puis complete par un script d'experiences pour automatiser les comparaisons et exporter les figures.

== Exo 1 : MultiLayer Perceptron

=== 1) Donnees

Le dataset Fashion-MNIST contient :
- 60 000 images d'apprentissage de taille $28 times 28$,
- 10 000 images de test de taille $28 times 28$,
- 10 classes (vetements et accessoires).

Le jeu de test est necessaire pour evaluer la capacite de generalisation. Le jeu d'apprentissage sert a ajuster les poids du reseau, tandis que le jeu de test mesure la performance sur des donnees jamais vues.

=== 2) Definition et parametrage du modele

Fonctions des couches principales :
- `Flatten` transforme une image $28 times 28$ en vecteur de 784 valeurs.
- `Dense` realise une transformation lineaire suivie d'une activation non-lineaire.

Activation utilisee :
- `relu` dans les couches cachees.

Parametres importants de `model.compile` :
- `optimizer`: ici Adam avec un learning rate configurable,
- `loss`: `SparseCategoricalCrossentropy(from_logits=True)`,
- `metrics`: `accuracy`.

Role de `model.fit` :
- lancer l'entrainement sur plusieurs epoques,
- mettre a jour les poids,
- retourner l'historique (`loss`, `accuracy`, etc.) pour l'analyse.

=== 3) Training et evolution de la loss

On observe une baisse progressive de la loss sur les premieres epoques, ce qui signifie que le modele apprend des representations plus discriminantes.

#figure(
  image("Capture_ecran_TP2_Classification/loss_mlp_baseline.png", width: 88%),
  caption: [Evolution de la loss - MLP baseline]
)

=== 4) Accuracy sur la base de test

Dans cette execution rapide (2 epoques, sous-ensemble de donnees), le MLP baseline atteint :

```txt
Test accuracy: 0.8333 (83.33%)
```

=== 5) Prediction et role du softmax

Le `softmax` convertit les logits du reseau en probabilites normalisees qui somment a 1. Chaque sortie represente la confiance du modele pour une classe.

== Impact hyperparametres (Exo 1 suite)

Configurations comparees :
- baseline: lr = 0.001, batch = 100,
- learning rate eleve: lr = 0.1,
- batch large: batch = 1000,
- reseau plus profond: ajout d'une couche dense de 64 neurones,
- CNN baseline (Exo 2).

#figure(
  image("Capture_ecran_TP2_Classification/tp2_accuracy_comparison.png", width: 92%),
  caption: [Comparaison des accuracies test]
)

#figure(
  image("Capture_ecran_TP2_Classification/tp2_time_comparison.png", width: 92%),
  caption: [Comparaison des temps d'entrainement]
)

Lecture des resultats :
- *Learning rate = 0.1* degrade fortement la stabilite et l'accuracy (53.07%).
- *Batch size = 1000* accelere l'entrainement mais penalise la performance (71.73%).
- *MLP plus profond* apporte un leger gain (83.90%).

== Exo 2 : Classification par CNN

Le CNN utilise :
- 1 couche `Conv2D(32, 3x3, relu)`,
- 1 couche `MaxPooling2D(2x2)`,
- 1 couche `Flatten`,
- 1 couche dense de sortie (10 logits).

Sur la meme execution rapide, le CNN atteint 83.70% d'accuracy, avec un temps d'entrainement un peu superieur au MLP.

=== Tableau de synthese

#table(
  columns: (1.7fr, 1fr, 1fr, 1fr),
  table.header(
    [*Modele*],
    [*Accuracy test*],
    [*Loss test*],
    [*Temps train (s)*],
  ),
  [MLP baseline], [83.33%], [0.4740], [1.26],
  [MLP lr = 0.1], [53.07%], [1.1105], [1.04],
  [MLP batch = 1000], [71.73%], [0.7624], [0.80],
  [MLP plus profond], [83.90%], [0.4828], [1.37],
  [CNN baseline], [83.70%], [0.4667], [1.61],
)

== Code ajoute / ameliore

Un script dedie a ete ajoute pour automatiser les experiences TP2 :

```txt
Codes/TP2_classification_experiments.py
```

Fonctions principales ajoutees :
- entrainement reproductible de plusieurs configurations,
- export automatique des courbes de loss,
- export des comparatifs accuracy/temps,
- export d'un fichier CSV de synthese.

Les figures du present rapport proviennent de :

```txt
Rapports/Capture_ecran_TP2_Classification/
```

== Conclusion

- Le MLP baseline donne deja une bonne base de performance.
- Un learning rate trop grand degrade fortement l'apprentissage.
- Augmenter le batch size peut reduire le temps, mais peut nuire a la qualite.
- Le CNN est competitif et plus adapte aux donnees image, surtout avec plus d'epoques.
