# Rapport TP1 : Reconnaissance de formes simples

**Date :** 15 janvier 2026  
**Anas DAGGAG - Jeremy DOUBLET - APP5 EIE**

---

## Objectif

Implémenter 3 méthodes de reconnaissance automatique de formes géométriques (cercle, triangle, octogone, carré, rectangle) basées sur différents descripteurs.

**Données :**
- 5 images de référence (00-04) : une par catégorie
- 22 images test (05-26) à classifier
- Classes : 0=cercle, 1=triangle, 2=octogone, 3=carré, 4=rectangle

---

## 1. Calcul des paramètres de forme

On a implémenté la fonction `myShapeCompute` qui extrait plusieurs descripteurs :

**Aire :** Nombre de pixels de l'objet (directement dans `stats[4]`)

**Périmètre :** Calculé via le code de Freeman
- Code de Freeman : encode les 8 directions possibles entre points consécutifs du contour
- On convertit ensuite en distance : 1 pour horizontal/vertical, √2 pour diagonal

**Périmètre :** Calculé via le code de Freeman
- Code de Freeman : encode les 8 directions possibles entre points consécutifs du contour
- On convertit ensuite en distance : 1 pour horizontal/vertical, √2 pour diagonal

**Compacité :** $C = \frac{4\pi A}{P^2}$ où A=aire, P=périmètre
- Proche de 1 pour un cercle, diminue pour les formes allongées

**Signature :** Distance de chaque point du contour au centre de gravité

**Moments de Hu :** 7 invariants géométriques (rotation, translation, échelle)

---

## 2. Méthode 1 : Classification par compacité

On compare la compacité de chaque image test avec les 5 références et on prédit la classe la plus proche.

### Résultats

```
prediction   : [2 0 2 4 1 1 3 1 0 3 3 3 3 1 4 1 1 1 2 3 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy : 68.2%
```

Valeurs observées :
- Cercles/Octogones : 0.90-0.95 → confusion fréquente
- Carrés : ~0.70-0.80
- Rectangles : ~0.50-0.70  
- Triangles : ~0.50-0.55

**Analyse :** Méthode simple mais limitée. Les octogones ressemblent trop aux cercles (forme régulière). Difficile aussi de distinguer carrés des rectangles selon leur orientation.

---

## 3. Méthode 2 : Classification par moments de Hu

On calcule la distance euclidienne entre les 7 moments de Hu de l'image test et ceux des références.

### Résultats

```
prediction   : [2 3 2 4 4 4 1 4 0 3 3 3 3 4 1 1 1 1 2 2 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy : 81.8%
```

**Analyse :** Meilleure performance. Les moments de Hu capturent bien la géométrie des formes et sont robustes aux rotations. Par contre, on a besoin d'images de référence et certaines confusions persistent (image 1 : octogone prédit comme carré, image 13 : triangle prédit comme rectangle).

---

## 4. Méthode 3 : Classification par signature

On analyse le nombre de maxima locaux dans la signature après filtrage gaussien :
- Cercle : variabilité très faible (signature quasi-constante)
- Triangle : 3 maxima (3 sommets)
- Carré/Rectangle : 4 maxima (4 coins)
- Octogone : 7-9 maxima (8 côtés)

### Exemples de signatures

**Cercle :** Signature constante, pas de variation

![Signature cercle](Signatures/signature_cercle.png)

**Triangle :** 3 maxima nets correspondant aux 3 sommets

![Signature triangle](Signatures/signature_triangle.png)

**Carré :** 4 maxima d'amplitudes égales (géométrie régulière)

![Signature carré](Signatures/signature_carree.png)

**Rectangle :** 4 maxima avec 2 amplitudes moins grandes 

![Signature rectangle](Signatures/signature_rectangle.png)

**Hexagone :** 6 maxima régulièrement espacés

![Signature hexagone](Signatures/signature_hexagone.png)

### Gestion des cas problématiques

**Problème 1 : Différenciation carré/rectangle**

Les deux formes ont 4 maxima. On les différencie en analysant le **rapport entre maxima** :

- **Carré :** Les 4 distances maximales sont identiques (4 coins équidistants du centre)
  - $\frac{max(maxima)}{min(maxima)} < 6$

- **Rectangle :** 2 distances différentes car les coins des côtés longs sont plus éloignés du centre que ceux des côtés courts
  - $\frac{max(maxima)}{min(maxima)} \geq 6$

Cette approche fonctionne bien car l'allongement du rectangle crée une différence significative entre les amplitudes des maxima.

**Problème 2 : Détection du cercle**

À cause de la pixelisation, un cercle peut présenter de nombreux petits maxima parasites. Le simple comptage ne suffit pas. On utilise donc l'**écart-type relatif** de la signature :

$$\frac{\sigma_{signature}}{\mu_{signature}} < 0.02$$

Pour un cercle, même pixellisé, la distance au centre varie très peu (signature quasi-constante). Ce critère permet de distinguer un cercle d'un octogone qui a aussi une signature relativement régulière mais avec 8 variations nettes.


### Résultats

```
prediction   : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy : 100%
```

**Analyse :** Performance parfaite ! Le comptage des maxima permet d'identifier précisément le nombre de sommets. Le filtrage gaussien avec $\sigma = \frac{longueur\_contour}{128}$ élimine bien le bruit de pixellisation. Avantage majeur : pas besoin d'images de référence.

---

## Synthèse

| Méthode | Accuracy | Avantages | Limites |
|---------|----------|-----------|---------|
| Compacité | 68.2% | Rapide, intuitive | Confusion cercle/octogone |
| Moments de Hu | 81.8% | Robuste aux rotations | Besoin de références |
| Signature | **100%** | Précise, sans référence | Sensible au filtrage |

---

## Conclusion

La méthode par signature s'est révélée la plus efficace avec 100% de bonnes classifications. Le comptage des maxima locaux permet d'identifier directement le nombre de sommets, ce qui est très adapté aux formes géométriques simples. 

Les moments de Hu donnent de bons résultats (81.8%) et seraient probablement meilleurs avec plus d'images de référence. La compacité seule reste trop limitée pour discriminer des formes proches comme cercle/octogone.

Pour une application réelle, on combinerait plusieurs descripteurs (compacité + signature) pour gagner en robustesse.

