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

$$\text{Compacité} = \frac{4*π*Aire}{p^2}$$

où P = périmètre et A = aire.

**Propriété :** La compacité est très proche de 1 pour un cercle et diminue pour les formes allongées ou irrégulières.

```python
compacite = 4*np.pi*aire/(perimetre**2) 
```

**Vérification pour un cercle :**
- Cercle parfait : $C = \frac{(2\pi r)^2}{\pi r^2} = 4\pi \approx 0.91$
- Dans nos résultats : compacité des cercles ≈ 0.93-0.91  (légère pixellisation)

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
    
    # Périmètre et compacité
    freeman_code = myFreemanCode(objet)
    perimetre = myPerimeter(freeman_code)
    compacite = 4*np.pi*aire/(perimetre**2) 
    
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
- **Cercles** : < 0.90
- **Octogones** : < 0.94 (très proche des cercles)
- **Carrés** : < 0.56
- **Triangles** : < 0.50 
- **Rectangles** : < 0.70 

### 2.4. Commentaires

**Points positifs :**

-  Triangles très bien reconnus (compacité élevée la plus basse)

**Limites observées :**
-  **Confusion cercle ↔ octogone** : compacités très proches (~0.90-0.95)
-  **Confusion carré ↔ rectangle** : même chose que pour certain rectangle / carré
---

## Question 3 : Fonction `myHuMomentsAnalysis` - Classification par moments de Hu

### 3.1. Principe

Les moments de Hu sont 7 invariants calculés à partir des moments centraux, invariants par :
- Translation
- Rotation  
- Changement d'échelle

Classification par plus proche voisin dans l'espace des moments de Hu (distance euclidienne).

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

MOMENT DE HU :
prediction   :   [2 3 2 4 4 4 1 4 0 3 3 3 3 4 1 1 1 1 2 2 0 0]
ground truth :   [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy: 0.8181818181818182

### 3.4. Commentaires

**Performance élevée  : 82%**

**Observations :**

**Points positifs :**

- Les écart entre chaque moments de Hu des différentes formes sont très grand
- peu importe le sens de la forme et la taille 

**Limites observées :**

- besoin d'image de référence 
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
-  **Triangles parfaitement détectés** (3 pics nets)
-  Méthode intuitive et interprétable géométriquement
- Pas besoin d'image de référence
**Limites observées :**
-  **Confusion carré ↔ rectangle** : même nombre de maxima (4), la variance seule ne suffit pas toujours
-  **Cercles mal classés** : certains identifiés comme "autre" ou "carré" à cause du bruit

**Améliorations possibles :**
Analyser l'amplitude des maxima, pas seulement leur nombre
---

## Synthèse comparative des méthodes

| Méthode | Accuracy | Avantages | Inconvénients |
|---------|----------|-----------|---------------|
| **Compacité** | **68%** | Simple et rapide<br>Bon pour formes régulières<br> besoin d'image de référence | Confusion cercle/octogone<br> Confusion carré/rectangle |
| **Moments de Hu** | **82%** | A compléter  | Besoin d'image de référence<br>|
| **Signature** | **100 %** |  Pas besoin d'image de référence<br> Intuitive géométriquement<br> Excellente pour triangles<br> Signature des formes propres |  Sensible au filtrage<br> Confusion sur les formes à 4 côtés (ajout d'une méthode de différentiation)<br> Besoin d'un dataset d'image de bonne qualité |

---

## Conclusion

Le TP a permis de découvrir une première approche du traitement d'image et d'analyser différente  méthodes de reconnaissance de formes géométriques :


**Enseignements :**
- Les descripteurs simples (compacité) sont souvent plus efficaces que les descripteurs complexes (moments de Hu) pour des formes géométriques basiques
- L'analyse de signature est puissante mais nécessite un bon prétraitement (filtrage)
- Aucune méthode seule n'atteint une performance parfaite
