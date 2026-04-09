# Prompt IA - Refaire entièrement Projet.ipynb pour le TP "Reconnaissance de panneaux"

Tu es un assistant expert en vision par ordinateur (traitement d'images) et en rédaction de notebooks pédagogiques.

## Mission
Réécris entièrement le notebook `Projet.ipynb` pour qu'il corresponde au sujet **Reconnaissance de panneaux de signalisation**.

- Remplace **tout** le contenu actuel (texte, code, graphiques, explications).
- Ne garde rien du tutoriel Fashion-MNIST/TensorFlow existant.
- Le notebook final doit être en **français**, clair, exécutable de haut en bas, et structuré comme un rendu de TP/projet d'ingénieur.

## Contexte du sujet
Le projet porte sur la reconnaissance automatique de panneaux de signalisation.

- Dataset principal: **GTSRB** (German Traffic Sign Recognition Benchmark).
- Alternative possible: dataset Kaggle de panneaux (si nécessaire).

## Contraintes obligatoires
Le notebook doit suivre exactement ces 4 parties:

1. **Présentation du jeu de données**
2. **Méthodes de reconnaissance de formes (approches classiques)**
3. **Méthode supervisée conçue et entraînée par nous**
4. **Utilisation d'un modèle pré-entraîné existant (transfer learning)**

Pour chaque partie: inclure **objectif**, **code**, **résultats**, **interprétation**.

## Structure attendue du notebook
Fournis le notebook complet cellule par cellule (Markdown + code Python), dans l'ordre d'exécution, avec:

- cellule d'installation optionnelle des dépendances si besoin
- imports propres
- seed fixée pour reproductibilité
- chargement des données (avec chemins clairement paramétrables)
- exploration des données:
  - nombre d'images
  - nombre de classes
  - distribution des classes
  - exemples d'images
- prétraitement:
  - redimensionnement
  - normalisation
  - split train/validation/test
  - gestion du déséquilibre si pertinent

### Partie 2 (classique)
- extraction de descripteurs (ex: HOG, couleur, contours)
- classifieur classique (ex: SVM, Random Forest ou KNN)
- métriques et limites de l'approche

### Partie 3 (supervisé maison)
- architecture CNN expliquée
- entraînement
- courbes loss/accuracy
- évaluation complète:
  - accuracy
  - precision
  - recall
  - F1-score
  - matrice de confusion
- analyse d'erreurs (exemples mal classés + hypothèses)
- au moins 3 expériences d'ablation (modifier/retirer des composants)
- tableau comparatif des expériences

### Partie 4 (modèle pré-entraîné)
- choix du backbone (ex: MobileNetV2 ou ResNet)
- stratégie de fine-tuning (couches gelées/dégelées)
- comparaison chiffrée avec le CNN maison

### Fin
- conclusion claire: résultats, limites, pistes d'amélioration
- cellule "Résumé final" avec:
  - meilleur modèle
  - meilleures métriques
  - temps d'entraînement

## Exigences de qualité
- Code robuste (gestion simple des erreurs de chemins/données manquantes).
- Noms de variables/fonctions explicites.
- Commentaires utiles mais concis.
- Graphiques lisibles (titres, axes, légendes).
- Pas de résultats inventés:
  - si exécution impossible, écrire "résultat attendu" et laisser un emplacement à compléter.

## Format de réponse attendu
Réponds en 3 blocs:

1. Plan rapide du notebook (sections + objectif de chaque section)
2. Notebook complet cellule par cellule, prêt à coller
3. Checklist de conformité au sujet (critère -> oui/non)

## Contrainte finale importante
Le contenu produit doit être **spécifique au sujet panneaux de signalisation** et non générique.
