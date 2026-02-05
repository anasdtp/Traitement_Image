# Rapport TP1 : Reconnaissance de formes simples

**Date :** 15 janvier 2026  
**Anas DAGGAG - Jeremy DOUBLET - APP5 EIE**

---

## Introduction

**Objectif du TP :** Implémenter des méthodes de reconnaissance de formes géométriques simples (cercle, triangle, octogone, carré, rectangle) en utilisant différents descripteurs de forme.

**Données :**
- Images 00 à 04 : 5 exemples de référence (un par catégorie)
- Images 05 à 26 : 22 images de test à classifier
- Catégories : 0=cercle, 1=triangle, 2=octogone, 3=carré, 4=rectangle

---

## Question 1 : Fonction `myShapeCompute` - Calcul des paramètres de forme

### 1.1. Calcul de l'aire

L'aire correspond au nombre de pixels de l'objet, disponible directement dans `stats[4]` retourné par `cv2.connectedComponentsWithStats`.

```python
aire = stats[4]  # Nombre de pixels à 1 dans l'objet
```

### 1.2. Calcul du périmètre

Le périmètre est calculé à partir du code de Freeman du contour.

#### a) Code de Freeman (`myFreemanCode`)

Encode chaque déplacement entre deux points consécutifs du contour selon 8 directions :

```
  3   2   1
   \  |  /
 4 - · - 0
   /  |  \
  5   6   7
```

**Implémentation :**

```python
def mySignature(objet):
    contour,_ = cv2.findContours(objet,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt=contour[0]
    ch=np.mean(cnt[:, 0, 0])  # Centre de gravité x
    cw=np.mean(cnt[:, 0, 1])  # Centre de gravité y
    signature=[]
   
    for point in cnt:
        x = point[0, 0]
        y = point[0, 1]
        distance = np.sqrt((x - ch)**2 + (y - cw)**2)
        signature.append(distance)
    
    return(signature)
```

#### **`myFreemanCode(objet)`**
Calcule le code de Freeman pour encoder le contour (8 directions possibles).

**Implémentation :**

```python
Freeman = np.array([[3, 2, 1], [4, -1, 0], [5, 6, 7]])

def myFreemanCode(objet):
    contour,_ = cv2.findContours(objet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contour[0]
    p0 = cnt[0]
    code = []
    
    for p in range(1, len(cnt)):
        p1 = cnt[p]
        # Calcul du vecteur de déplacement
        dx = p1[0, 0] - p0[0, 0]
        dy = p1[0, 1] - p0[0, 1]
        # Normalisation à -1, 0, ou 1
        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        # Indexation dans la matrice Freeman (dy+1, dx+1)
        code.append(Freeman[dy+1, dx+1])
        p0 = p1
    
    return code
```

#### b) Calcul du périmètre (`myPerimeter`)

À partir du code de Freeman, on calcule le périmètre en sommant les distances :
- **Directions horizontales/verticales** (0,2,4,6) : distance = 1
- **Directions diagonales** (1,3,5,7) : distance = √2

```python
def myPerimeter(freeman_code):
    perim = 0.
    for p in range(0, len(freeman_code)):
        if freeman_code[p] % 2 == 0:  # Direction paire
            perim += 1.0
        else:  # Direction impaire
            perim += np.sqrt(2)
    return perim
```

### 1.3. Calcul de la compacité

La compacité est un descripteur de forme défini par :

$$\text{Compacité} = \frac{P^2}{A}$$

où P = périmètre et A = aire.

**Propriété :** La compacité est minimale pour un cercle (≈ 4π ≈ 12.57) et augmente pour les formes allongées ou irrégulières.

```python
compacite = (perimetre**2) / aire
```

**Vérification pour un cercle :**
- Cercle parfait : $C = \frac{(2\pi r)^2}{\pi r^2} = 4\pi \approx 12.57$
- Dans nos résultats : compacité des cercles ≈ 13.4-13.9 (légère pixellisation)

### 1.4. Fonction complète `myShapeCompute`

```python
def myShapeCompute(objet, stats):
    S = []  # Liste des paramètres
    
    # Aire
    aire = stats[4]
    
    # Pseudo-rectangularité
    pseudorect = aire / (stats[2] * stats[3])
    
    # Moments et moments de Hu
    moments = cv2.moments(objet)
    hu_moments = cv2.HuMoments(moments)
    for i in range(0, 7):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log(abs(hu_moments[i]))
    
    # Périmètre et compacité
    freeman_code = myFreemanCode(objet)
    perimetre = myPerimeter(freeman_code)
    compacite = (perimetre**2) / aire
    
    # Signature
    sig = mySignature(objet)
    
    # Stockage des paramètres
    S.append(aire)          # [0]
    S.append(perimetre)     # [1]
    S.append(compacite)     # [2]
    S.append(pseudorect)    # [3]
    S.append(sig)           # [4]
    S.append(moments)       # [5]
    S.append(hu_moments)    # [6]
    
    return S
```

---

## Question 2 : Fonction `myCompacityAnalysis` - Classification par compacité

### 2.1. Principe

Pour chaque forme à tester :
1. Calculer sa compacité
2. Comparer avec la compacité des 5 exemples
3. Prédire la catégorie de l'exemple le plus proche (distance minimale)
4. Comparer avec la vérité terrain (ground_truth)

### 2.2. Implémentation

```python
def myCompacityAnalysis(param_test, param_ex):
    nb_objets = len(param_test)
    nb_ex = len(param_ex)
    prediction = np.zeros(shape=(nb_objets), dtype=np.uint8)
    TP = 0  # True Positives
    compacite = []
    
    for i in range(nb_objets):
        comp_test = param_test[i][2]  # Compacité à l'indice 2
        compacite.append(comp_test)
        
        # Recherche du plus proche voisin
        min_dist = float('inf')
        best_match = 0
        
        for j in range(nb_ex):
            comp_ex = param_ex[j][2]
            dist = abs(comp_test - comp_ex)
            
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        prediction[i] = best_match
        
        # Vérification avec ground truth
        if prediction[i] == ground_truth[i]:
            TP += 1
    
    accuracy = TP / len(prediction)
    return prediction, accuracy
```

### 2.3. Résultats

```
COMPACITÉ :
prediction    : [2 0 2 4 1 1 3 1 0 3 3 3 3 1 4 1 1 1 2 3 0 0]
ground truth  : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy      : 68.2%
```

**Valeurs de compacité observées :**
- **Cercles** : 13.4 - 13.9 (proche de 4π)
- **Octogones** : 13.2 - 13.7 (très proche des cercles)
- **Carrés** : 15.3 - 18.3
- **Triangles** : 23.3 - 23.6 (formes angulaires)
- **Rectangles** : 22.1 - 24.5 (formes allongées)

### 2.4. Commentaires

**Points positifs :**
- ✅ Bonne séparation entre formes arrondies (cercle/octogone) et formes angulaires (triangle/rectangle)
- ✅ Triangles très bien reconnus (compacité élevée distinctive)
- ✅ Méthode simple et rapide

**Limites observées :**
- ⚠️ **Confusion cercle ↔ octogone** : compacités très proches (~13), les octogones réguliers sont presque circulaires
- ⚠️ **Confusion carré ↔ rectangle** : difficile de les différencier uniquement par compacité
- ⚠️ La compacité est sensible à la pixellisation et au bruit du contour

**Performance :** 68.2% - La meilleure des trois méthodes testées.

---

## Question 3 : Fonction `myHuMomentsAnalysis` - Classification par moments de Hu

### 3.1. Principe

Les moments de Hu sont 7 invariants calculés à partir des moments centraux, invariants par :
- Translation
- Rotation  
- Changement d'échelle

Classification par plus proche voisin dans l'espace des moments de Hu (distance euclidienne).

### 3.2. Implémentation

### 3.2. Implémentation

```python
def myHuMomentsAnalysis(param_test, param_ex):
    nb_objets = len(param_test)
    nb_ex = len(param_ex)
    prediction = np.zeros(shape=(nb_objets), dtype=np.uint8)
    TP = 0
    
    for i in range(nb_objets):
        hu_test = param_test[i][6]  # Moments de Hu à l'indice 6
        
        # Recherche du plus proche voisin
        min_dist = float('inf')
        best_match = 0
        
        for j in range(nb_ex):
            hu_ex = param_ex[j][6]
            # Distance euclidienne entre les vecteurs de 7 moments
            dist = np.linalg.norm(hu_test - hu_ex)
            
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        prediction[i] = best_match
        
        if prediction[i] == ground_truth[i]:
            TP += 1
    
    accuracy = TP / len(prediction)
    return prediction, accuracy
```

### 3.3. Résultats

```
MOMENTS DE HU :
prediction    : [2 1 0 1 3 2 0 3 1 3 4 2 0 0 1 2 1 1 0 3 3 2]
ground truth  : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy      : 22.7%
```

### 3.4. Commentaires

**Performance très faible : 22.7%**

**Observations :**
- ❌ Nombreuses confusions entre toutes les catégories
- ❌ Les moments de Hu semblent trop sensibles aux variations mineures
- ❌ Peu discriminants pour des formes géométriques simples

**Explications possibles :**
1. **Sensibilité au bruit** : Petites variations de contour amplifient les différences
2. **Trop d'invariance** : L'invariance rotation peut masquer des différences importantes
3. **Inadapté aux formes simples** : Les moments de Hu sont plus adaptés aux formes complexes ou organiques
4. **Problème de normalisation** : Les échelles des 7 moments sont très différentes

**Amélioration possible** : Normaliser ou pondérer différemment les 7 moments, ou utiliser seulement certains moments plus discriminants.

---

## Question 4 : Fonction `mySignatureAnalysis` - Classification par signature

### 4.1. Principe

La **signature** d'une forme est la distance de chaque point du contour au centre de gravité. 

#### Calcul de la signature (`mySignature`)

```python
def mySignature(objet):
    contour,_ = cv2.findContours(objet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contour[0]
    
    # Centre de gravité
    ch = np.mean(cnt[:, 0, 0])  # Coordonnée x moyenne
    cw = np.mean(cnt[:, 0, 1])  # Coordonnée y moyenne
    
    signature = []
    for point in cnt:
        x = point[0, 0]
        y = point[0, 1]
        # Distance euclidienne au centre
        distance = np.sqrt((x - ch)**2 + (y - cw)**2)
        signature.append(distance)
    
    return signature
```

### 4.2. Analyse des maxima locaux

Le nombre de **maxima locaux** (pics) dans la signature correspond au nombre de "coins" ou sommets de la forme :
- **Cercle** : signature quasi-constante → peu de maxima (≤2)
- **Triangle** : 3 sommets → 3 maxima
- **Carré** : 4 sommets → 4 maxima
- **Rectangle** : 4 sommets → 4 maxima  
- **Octogone** : 8 sommets → 7-9 maxima (avec le lissage)

### 4.3. Implémentation

```python
def mySignatureAnalysis(param_test):
    nb_objets = len(param_test)
    prediction = np.zeros(shape=(nb_objets), dtype=np.uint8)
    TP = 0
    
    for i in range(nb_objets):
        signature = param_test[i][4]  # Signature à l'indice 4
        
        # Conversion en liste si nécessaire
        if not isinstance(signature, list):
            signature_list = list(signature)
        else:
            signature_list = signature
        
        # Normalisation : rotation pour commencer au minimum
        signature_list = np.roll(signature_list, 
                                -signature_list.index(min(signature_list)))
        signature = np.asarray(signature_list)
        
        # Filtrage gaussien pour éliminer le bruit
        longueur_contour = len(signature)
        sigma = longueur_contour / 128
        signature = sc.ndimage.gaussian_filter(signature, sigma, order=0)
        
        # Détection des maxima locaux
        maxima = argrelextrema(signature, np.greater)[0]
        nb_maxima = len(maxima)
        
        # Classification selon le nombre de maxima
        if nb_maxima <= 2:
            prediction[i] = CERCLE
        elif nb_maxima == 3:
            prediction[i] = TRIANGLE
        elif nb_maxima == 4:
            # Différenciation carré/rectangle par variance
            distances = [signature[maxima[k]] for k in range(nb_maxima)]
            variance = np.var(distances)
            if variance < np.mean(distances) * 0.1:
                prediction[i] = CARRE  # Distances homogènes
            else:
                prediction[i] = RECTANGLE  # Distances variées
        elif 7 <= nb_maxima <= 9:
            prediction[i] = OCTOGONE
        else:
            prediction[i] = AUTRE
        
        if prediction[i] == ground_truth[i]:
            TP += 1
    
    accuracy = TP / len(prediction)
    return prediction, accuracy
```

### 4.4. Résultats

```
SIGNATURE :
prediction    : [2 2 2 3 3 3 3 3 5 3 3 3 3 1 1 1 1 1 2 5 5 3]
ground truth  : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy      : 63.6%
```

### 4.5. Commentaires

**Performance moyenne : 63.6%**

**Points positifs :**
- ✅ **Triangles parfaitement détectés** (3 pics nets)
- ✅ Méthode intuitive et interprétable géométriquement
- ✅ Octogones bien identifiés grâce aux 8 sommets

**Limites observées :**
- ⚠️ **Confusion carré ↔ rectangle** : même nombre de maxima (4), la variance seule ne suffit pas toujours
- ⚠️ **Cercles mal classés** : certains identifiés comme "autre" ou "carré" à cause du bruit
- ⚠️ **Sensibilité au filtrage** : le paramètre σ influence fortement le nombre de maxima détectés
- ⚠️ Pixellisation crée des irrégularités dans la signature

**Améliorations possibles :**
1. Ajuster le paramètre de filtrage gaussien (σ)
2. Utiliser le rapport hauteur/largeur pour différencier carré/rectangle
3. Analyser l'amplitude des maxima, pas seulement leur nombre
4. Combiner avec d'autres descripteurs (compacité, moments)

---

## Synthèse comparative des méthodes

| Méthode | Accuracy | Avantages | Inconvénients |
|---------|----------|-----------|---------------|
| **Compacité** | **68.2%** | ✅ Simple et rapide<br>✅ Bon pour formes régulières<br>✅ Interprétable | ⚠️ Confusion cercle/octogone<br>⚠️ Confusion carré/rectangle |
| **Moments de Hu** | **22.7%** | ✅ Invariants théoriques robustes<br>✅ Applicable formes complexes | ❌ Peu discriminant formes simples<br>❌ Sensible au bruit<br>❌ Difficile à interpréter |
| **Signature** | **63.6%** | ✅ Intuitive géométriquement<br>✅ Excellente pour triangles<br>✅ Détecte le nb de sommets | ⚠️ Sensible au filtrage<br>⚠️ Confusion 4 côtés<br>⚠️ Pixellisation |

---

## Conclusion

Le TP a permis d'implémenter avec succès trois méthodes de reconnaissance de formes géométriques :

**Classement des performances :**
1. 🥇 **Compacité : 68.2%** - Meilleure méthode globalement
2. 🥈 **Signature : 63.6%** - Bonne performance, très efficace pour les triangles  
3. 🥉 **Moments de Hu : 22.7%** - Inadaptés aux formes géométriques simples

**Enseignements :**
- Les descripteurs simples (compacité) sont souvent plus efficaces que les descripteurs complexes (moments de Hu) pour des formes géométriques basiques
- L'analyse de signature est puissante mais nécessite un bon prétraitement (filtrage)
- Aucune méthode seule n'atteint une performance parfaite

**Perspectives d'amélioration :**
1. **Fusion de méthodes** : Vote majoritaire entre les 3 approches
2. **Descripteurs additionnels** : rapport hauteur/largeur, circularité
3. **Apprentissage supervisé** : SVM ou k-NN sur l'ensemble des descripteurs
4. **Plus d'exemples** : 5 exemples par classe sont insuffisants pour capturer la variabilité