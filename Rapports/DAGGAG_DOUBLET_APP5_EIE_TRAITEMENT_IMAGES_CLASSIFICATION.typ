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

Le modèle MLP (Multi-Layer Perceptron) est un réseau de neurones entièrement connecté, dans lequel chaque neurone d’une couche est relié à tous les neurones de la couche suivante. Dans ce cas, l’image 28×28 est d’abord aplatie en un vecteur 1D de 784 valeurs grâce à la couche Flatten, puis traitée par plusieurs couches Dense. Ce type de modèle est simple et efficace, mais il ne prend pas en compte la structure spatiale de l’image.


=== 3) Training et evolution de la loss

On observe une baisse progressive de la loss sur les premieres epoques, ce qui signifie que le modele apprend des representations plus discriminantes.


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


Lecture des resultats :
- *Learning rate = 0.1* degrade fortement la stabilite et l'accuracy (53.07%).
- *Batch size = 1000* accelere l'entrainement mais penalise la performance (71.73%).
- *MLP plus profond* apporte un leger gain (83.90%).

le CNN (Convolutional Neural Network) est spécialement conçu pour le traitement d’images. Il utilise des couches de convolution (Conv2D) qui détectent automatiquement des motifs locaux comme les bords, formes ou textures, suivies de couches de pooling (MaxPooling2D) qui réduisent la taille des données tout en conservant les informations importantes. Ainsi, le CNN exploite mieux l’organisation des pixels dans l’image et offre généralement de meilleures performances que le MLP pour la classification d’images.

Sur la meme execution rapide, le CNN atteint 83.70% d'accuracy, avec un temps d'entrainement un peu superieur au MLP.

=== Tableau de synthese


Différent test ont été réalisé pour comparer le fonctionnement des modèles #image("/assets/image.png")

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
  [CNN], [83.70%], [0.4667], [1.61],
)

== Conclusion

- Le MLP baseline donne deja une bonne base de performance.
- Un learning rate trop grand degrade fortement l'apprentissage.
- Augmenter le batch size peut reduire le temps, mais peut nuire a la qualite.
- Le CNN fonctionne mieux et est plus adapté avec beaucoup d'image et avec plus d'epoques.


Nous pouvons voir aussi que le modèle donne toujours des mauvais résultat et donc qu'il faut quand même vérifier les résultat. 

#image("/assets/image-1.png")
 
Dans ce cas le modèle confond la classe du sac au lieux de la classe sneaker