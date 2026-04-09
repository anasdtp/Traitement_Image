#set page(margin: (x: 2.2cm, y: 2.2cm))
#set par(justify: true)
#set text(lang: "fr")

= Rapport TP3 : Detection de mouvement et flot optique

*Date :* 19 Mars 2026\
*Anas DAGGAG - Jeremy DOUBLET - APP5 EIE*

== Objectif

Ce TP traite la detection de mouvement dans une video fixe (`campus.mp4`) avec :
- soustraction de fond,
- normalisation photometrique,
- mise a jour dynamique du fond,
- morphologie mathematique,
- comptage par composantes connexes,
- estimation de flot optique dense.

Les codes ont ete completes dans :

```txt
Codes/motion.py
Codes/TP3_mouvement.py
```

== Exo 1 : Soustraction de fond

La soustraction de fond implementee est basee sur :

$ F(x) = cases(1 "si" abs(I_t(x)-B_t(x)) > s, 0 "sinon") $

avec :
- $I_t$ l'image courante,
- $B_t$ l'image de fond,
- $s$ un seuil (ici $s = 20$).

#figure(
	image("Capture_ecran_TP3_Mouvement/frame_0030_gray.png", width: 48%),
	caption: [Image en niveaux de gris (frame 30)]
)

#figure(
	image("Capture_ecran_TP3_Mouvement/frame_0030_foreground_raw.png", width: 48%),
	caption: [Foreground brut (sans morphologie)]
)

Commentaires :
- des fantomes apparaissent lorsque le fond evolue lentement ou que des objets se deplacent lentement,
- les changements d'eclairage degradent fortement un seuillage simple,
- le foreground brut contient beaucoup de bruit ponctuel.

== Exo 2 : Normalisation photometrique

La normalisation implementee dans `photometric_normalization(image)` est :

$ I_n(x) = (I(x)-mu)/sigma $

suivie d'une remise a l'echelle 8 bits pour le traitement OpenCV.

Cette normalisation compense principalement :
- les variations globales de luminosite,
- les changements de contraste global.

Limite : en cas de changements non-uniformes d'eclairage (ombres locales), cette approche est insuffisante.

== Exo 3 : Mise a jour du fond

Formule moyenne sur $N$ images :

$ B_t(x) = (1/N) sum_(k=0)^(N-1) I_(t-k)(x) $

Forme iterative utilisee :

$ B_t = alpha B_(t-1) + (1-alpha) I_t $

avec $alpha = 0.8$ dans notre implementation (`update_beackground_mean`).

Influence du parametre :
- grand $alpha$ : fond plus stable mais adaptation lente,
- petit $alpha$ : adaptation rapide mais risque d'integrer les objets mobiles dans le fond.

== Exo 4 : MOG2 / KNN (analyse)

MOG2/KNN sont des methodes plus robustes aux variations d'illumination et aux fonds dynamiques que la soustraction simple. Dans notre pipeline, la strategie moyenne temporelle + normalisation + morphologie donne deja un resultat exploitable, mais MOG2/KNN seraient preferables pour une scene plus complexe.

== Exo 5 : Comptage d'objets

Objectif : obtenir une composante connexe propre par objet mobile.

Pipeline retenu :
- `MORPH_OPEN` avec noyau elliptique 3x3 (suppression du bruit isole),
- `MORPH_CLOSE` avec noyau elliptique 9x9 (fusion locale des zones),
- filtrage par aire minimale (`MIN_AREA = 120`) avant comptage final.

#figure(
	image("Capture_ecran_TP3_Mouvement/frame_0120_foreground_morph.png", width: 48%),
	caption: [Foreground apres morphologie (frame 120)]
)

#figure(
	image("Capture_ecran_TP3_Mouvement/frame_0120_annotated.png", width: 48%),
	caption: [Objets detectes et boites englobantes]
)

#figure(
	image("Capture_ecran_TP3_Mouvement/tp3_counts_comparison.png", width: 90%),
	caption: [Comptage d'objets : sans/avec morphologie]
)

Observation : le comptage sans morphologie est tres bruit e (nombre de composantes beaucoup trop grand), alors que le pipeline morphologique stabilise fortement le nombre d'objets detectes.

== Exo 6 : Flot optique

Le flot optique estime le deplacement apparent des pixels entre deux images consecutives. Ici, on utilise un flot optique dense (`cv2.calcOpticalFlowFarneback`).

Pour chaque composante connexe filtree, on calcule le flot moyen :
- calcul de la norme du vecteur flot pour chaque pixel,
- moyenne de cette norme sur les pixels de la composante.

#figure(
	image("Capture_ecran_TP3_Mouvement/frame_0220_flow.png", width: 90%),
	caption: [Visualisation HSV du flot optique dense]
)

#figure(
	image("Capture_ecran_TP3_Mouvement/tp3_mean_optical_flow.png", width: 90%),
	caption: [Evolution du flot optique moyen]
)

== Resultats exportes

Les sorties automatiques sont disponibles dans :

```txt
Rapports/Capture_ecran_TP3_Mouvement/
```

Fichiers principaux :
- captures de frames (`frame_0030_*`, `frame_0120_*`, `frame_0220_*`),
- `tp3_counts_comparison.png`,
- `tp3_mean_optical_flow.png`,
- `tp3_summary.csv`.



Pour essayer la detection sur d'autres modèle il faut maintenant jouer sur les valeurs de : THRESHOLD = 10
ALPHA_BG = 0.7
MIN_AREA = 100

Voici le résultat pour les autres vidéo : 
Vidéo lego 

#image("/assets/image-2.png")

THRESHOLD = 20
ALPHA_BG = 0.2
MIN_AREA = 10
MAX_FRAMES = 260


Cette vidéo est très difficile à analyser car les couleurs se raproche beaucoup et la vidéo est très courte 

Vidéo rouen 

#image("/assets/image-3.png")

J'ai trouvé cette image intéréssante car nous pouvons voir que le réglage du paramètre pour retiré les ombres est très important, sur l'image ont peut voir que les ombre du véicule sont considéré comme du mouvement 

Vidéo sheerbrooke 

#image("/assets/image-4.png")


Cette vidéo est très facile à analyser car les couleurs sont très différentes et la vidéo est de bonne qualité (très bonne luminosité)

THRESHOLD = 10
ALPHA_BG = 0.5
MIN_AREA = 10
MAX_FRAMES = 260

Vidéo stmarc_video

#image("/assets/image-5.png")

Sur cette vidéo l'élement interessant est l'arbre, ont peut voir que les feuilles sont considéré comme du mouvement 

#image("/assets/image-6.png")

THRESHOLD = 20
ALPHA_BG = 0.2
MIN_AREA = 10
MAX_FRAMES = 260

Avec ces paramètre nous arrivons à retirer les bruits des feuilles de l'abre ce qui montre vraiment l'importance du choix des paramètre de filtrage 

== Conclusion

- La soustraction de fond simple fonctionne mais est sensible a l'eclairage et au bruit.
- La normalisation photometrique et la mise a jour temporelle du fond ameliorent la robustesse.
- La morphologie est essentielle pour stabiliser le comptage d'objets.
- Le flot optique moyen par composante apporte une information dynamique utile pour qualifier le mouvement, au-dela d'une simple detection binaire.
- Pour une détection la plus optimal le choix des paramètre est très important et influx énormément en fonction de la vidéo traité 