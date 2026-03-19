#set page(margin: (x: 2.2cm, y: 2.2cm))
#set par(justify: true)
#set text(lang: "fr")

= Rapport TP1 : Reconnaissance de formes simples

#figure(
  image("image.png", width: 100%),
  caption: [Jeu d'images utilise pour la reconnaissance]
)

*Date :* 12 Fevrier 2026\
*Anas DAGGAG - Jeremy DOUBLET - APP5 EIE*

== Objectif

Implementer trois methodes de reconnaissance automatique de formes geometriques (cercle, triangle, octogone, carre, rectangle) basees sur differents descripteurs.

*Donnees :*
- 5 images de reference (00-04), une par categorie.
- 22 images test (05-26) a classifier.
- Classes : 0 = cercle, 1 = triangle, 2 = octogone, 3 = carre, 4 = rectangle.

== Exo 1.1 : Compacite

=== Function myShapeCompute (dans shape.py)

La fonction prend en entree la forme a reconnaitre et calcule plusieurs parametres.

- *Aire* : nombre de pixels de l'objet. L'aire est obtenue dans `stats[4]`.
- *Perimetre* : longueur du contour.
  - Completer `myFreemanCode` pour deduire le code de Freeman depuis la chaine du contour obtenue via `cv2.findContours`.
  - Completer `myPerimeter` pour en deduire le perimetre.

Le code de Freeman encode les 8 directions possibles entre points consecutifs du contour. Pour chaque point, on calcule le vecteur `d = [dx, dy]` vers le point suivant, puis on normalise a -1, 0 ou 1, et on utilise la matrice de Freeman pour obtenir la direction (0 a 7).

Le perimetre se calcule ensuite en sommant les distances : 1 pour les directions horizontales/verticales (0, 2, 4, 6) et $sqrt(2)$ pour les diagonales (1, 3, 5, 7).

- *Compacite* : verifier qu'elle est maximale pour les ronds.

La compacite est calculee par :

$ C = (4 pi A) / P^2 $

avec $A$ l'aire et $P$ le perimetre. Elle est proche de 1 pour un cercle (forme la plus compacte) et diminue pour les formes allongees.

=== Function myCompacityAnalysis(param_test, param_ex)

La fonction completee :
- compare la compacite de chaque forme (`param_test`) avec les exemples (`param_ex`) ;
- renvoie la categorie pour chaque forme (0 : cercle, 1 : triangle, 2 : octogone, 3 : carre, 4 : rectangle) ;
- permet de commenter les resultats obtenus.

=== Resultats

```txt
prediction   : [2 0 2 4 1 1 3 1 0 3 3 3 3 1 4 1 1 1 2 3 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
compacite    : [0.94, 0.92, 0.94, 0.57, 0.53, 0.29, 0.74, 0.51, 0.91, 0.69, 0.69, 0.69, 0.8, 0.54, 0.57, 0.53, 0.53, 0.54, 0.95, 0.82, 0.91, 0.9]
Accuracy     : 68.2%
```

Rappel :

```txt
CERCLE = 0
TRIANGLE = 1
OCTOGONE = 2
CARRE = 3
RECTANGLE = 4
AUTRE = 5
```

Valeurs de compacite observees :
- Cercles/Octogones : 0.90-0.95, confusion frequente.
- Carres : environ 0.70-0.80.
- Rectangles : environ 0.50-0.70.
- Triangles : environ 0.50-0.55.

Les octogones ressemblent trop aux cercles, et il reste difficile de distinguer carres et rectangles.

== Exo 1.2 : Moments de Hu

=== Calcul des moments de Hu

Dans `myShapeCompute`, ajout du calcul des moments de Hu avec OpenCV.

On utilise d'abord `cv2.moments()` pour les moments geometriques (invariants a la translation), puis `cv2.HuMoments()` pour calculer les 7 moments de Hu, invariants a la translation, la rotation et l'echelle.

=== Function myHuMomentsAnalysis(param_test, param_ex)

La fonction completee :
- compare les 7 moments de Hu d'une forme test avec ceux des formes exemple ;
- renvoie la categorie de la forme ;
- permet d'analyser les confusions residuelles.

=== Resultats

```txt
prediction   : [2 3 2 4 4 4 1 4 0 3 3 3 3 4 1 1 1 1 2 2 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy     : 81.8%
```

Performance meilleure que la compacite. Les moments de Hu capturent bien la geometrie des formes et sont robustes aux rotations. En revanche, cette methode depend d'images de reference et certaines confusions persistent (octogone predit comme carre, triangle predit comme rectangle).

== Exo 1.3 : Signature

=== Function mySignature

La fonction est completee pour renvoyer la signature de l'objet.

=== Function mySignatureAnalysis(param_test)

La fonction est mise a jour pour analyser la signature et deduire la classe de chaque forme.

Regles observees :
- Cercle : variabilite tres faible (signature quasi constante).
- Triangle : 3 maxima (3 sommets).
- Carre/Rectangle : 4 maxima (4 coins).
- Octogone : 7 a 9 maxima (8 cotes).

=== Exemples de signatures

*Cercle* : signature constante, peu de variation.

#figure(
  image("Signatures/signature_cercle.png", width: 68%),
  caption: [Signature d'un cercle]
)

*Triangle* : 3 maxima nets correspondant aux 3 sommets.

#figure(
  image("Signatures/signature_triangle.png", width: 68%),
  caption: [Signature d'un triangle]
)

*Carre* : 4 maxima d'amplitudes egales (geometrie reguliere).

#figure(
  image("Signatures/signature_carree.png", width: 68%),
  caption: [Signature d'un carre]
)

*Rectangle* : 4 maxima avec 2 amplitudes plus faibles.

#figure(
  image("Signatures/signature_rectangle.png", width: 68%),
  caption: [Signature d'un rectangle]
)

*Hexagone* : 6 maxima regulierement espaces.

#figure(
  image("Signatures/signature_hexagone.png", width: 68%),
  caption: [Signature d'un hexagone]
)

=== Gestion des cas problematiques

*Probleme 1 : Differenciation carre/rectangle*

Les deux formes ont 4 maxima. On les differencie en analysant le rapport entre maxima :
- *Carre* : les 4 distances maximales sont identiques (coins equidistants du centre).
- *Rectangle* : deux distances differentes, car les coins des cotes longs sont plus eloignes du centre que ceux des cotes courts.

#figure(
  image("image-2.png", width: 30%),
  caption: [Cas carre]
)

#figure(
  image("image-3.png", width: 30%),
  caption: [Cas rectangle]
)

Cette approche fonctionne bien car l'allongement du rectangle cree une difference significative entre les amplitudes des maxima.

*Probleme 2 : Detection du cercle*

A cause de la pixelisation, un cercle peut presenter de nombreux petits maxima parasites. Le simple comptage ne suffit pas ; on utilise donc l'ecart-type relatif de la signature :

$ "sigma_rel" = sigma(r) / bar(r) $

#figure(
  image("image-4.png", width: 30%),
  caption: [Critere base sur l'ecart-type relatif]
)

Pour un cercle, meme pixelise, la distance au centre varie peu (signature quasi constante). Ce critere le distingue d'un octogone, qui presente des variations nettes.

=== Resultats

```txt
prediction   : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
ground truth : [2 2 2 4 4 4 4 4 0 3 3 3 3 1 1 1 1 1 2 5 0 0]
Accuracy     : 100%
```

Performance parfaite. Le comptage des maxima identifie precisement le nombre de sommets. Le filtrage gaussien avec $sigma = 5$ elimine bien le bruit de pixelisation. Avantage majeur : pas besoin d'images de reference.

== Synthese

#table(
  columns: (1.4fr, 0.8fr, 1.8fr, 1.8fr),
  align: (x, y) => if x == 1 { center } else { left },
  table.header(
    [*Methode*],
    [*Accuracy*],
    [*Avantages*],
    [*Inconvenients*],
  ),
  [*Compacite*],
  [*68%*],
  [Simple et rapide ; calcul peu couteux ; correct pour formes regulieres.],
  [Besoin d'images de reference ; confusion cercle/octogone ; confusion carre/rectangle.],
  [*Moments de Hu*],
  [*82%*],
  [Robuste aux rotations et changements d'echelle ; meilleure performance que la compacite ; capture bien la geometrie.],
  [Besoin d'images de reference ; calcul plus complexe ; confusions residuelles.],
  [*Signature*],
  [*100%*],
  [Pas besoin d'images de reference ; interpretation geometrique intuitive ; identification directe du nombre de sommets ; performance parfaite sur ce jeu de donnees.],
  [Sensible au choix du filtrage ; necessite un bon pretraitement ; parametre $sigma$ a ajuster selon la taille du contour.],
)

== Conclusion

- Les descripteurs simples (compacite) restent efficaces pour des formes geometriques basiques.
- L'analyse de signature est tres performante, mais demande un pretraitement de qualite.
- Aucune methode seule n'est universelle ; la combinaison de criteres reste la strategie la plus robuste.
