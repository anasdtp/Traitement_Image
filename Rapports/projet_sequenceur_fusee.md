# Projet Séquenceur - Fusée Expérimentale

## Vue d'ensemble

Conception d'une carte électronique séquenceur pour fusée expérimentale, gérant le cycle de vol depuis le décollage jusqu'à la récupération.

---

## Carte Séquenceur (Carte Principale)

### Microcontrôleur
- **Teensy 4.1** (microcontrôleur principal)

### Fonctions principales

#### 1. Détection du décollage
- **Méthode**: Prise jack (détection par séparation électrique)
- Reconnaissance automatique du décollage lors de la séparation

#### 2. Acquisition de données
- **IMU** (Inertial Measurement Unit)
- **Stockage**: Carte SD (via Teensy 4.1)
- Enregistrement continu des données de vol

#### 3. Gestion du parachute
- **Actionneur**: Servomoteur
- Contrôle du déploiement selon la séquence programmée

#### 4. Interface Homme-Machine (IHM)
- **Entrées**: 4 interrupteurs
- **Sorties**: 2 LEDs
- Interface simple pour configuration et monitoring

#### 5. Communication avec carte expérience
- **Signal**: Rising edge pour départ de l'expérience
- **Isolation**: Optocoupleur (résolution problème GND différent)

### Contraintes mécaniques
- **Format**: Carte ronde
- **Diamètre**: < 9.8 cm
- **Fixation**: 4 trous de montage

### Connecteurs
1. Connecteur IMU
2. Connecteur expérience (avec optocoupleur)
3. Connecteur servomoteur
4. Connecteur jack (détection décollage)

---

## Carte Expérience (Carte Secondaire)

### Architecture
- **Alimentation**: Indépendante de la carte séquenceur
- **Microcontrôleur**: Propre µC (non déterminé)
- **IMU**: Capteur dédié

### Interface avec séquenceur
- **Signal d'entrée**: Rising edge depuis carte séquenceur
- **Isolation**: Via optocoupleur (côté séquenceur)

### Statut
⚠️ **Carte non encore conçue/déterminée**

---

## Problématiques et Solutions

### Séparation des masses (GND)
- **Problème**: Les GND de la carte séquenceur et de la carte expérience sont différents
- **Raison**: Alimentations indépendantes
- **Solution**: Optocoupleur sur la carte séquenceur pour isolation galvanique
- **Avantage**: Communication sans liaison électrique directe

---

## Outils de conception

- **Logiciel**: KiCad (schématique et PCB)

---

## Architecture système

```
┌─────────────────────────────────────────────────────┐
│           CARTE SÉQUENCEUR (Teensy 4.1)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Jack Décollage] ──→ Détection décollage          │
│                                                     │
│  [IMU] ──→ Acquisition ──→ [SD Card]               │
│                                                     │
│  [IHM] ←→ 4 Switches + 2 LEDs                      │
│                                                     │
│  [Servomoteur] ←── Contrôle parachute              │
│                                                     │
│  [Optocoupleur] ──→ Signal départ (rising edge)    │
│                     │                               │
└─────────────────────┼───────────────────────────────┘
                      │
                      │ (Isolation galvanique)
                      │
┌─────────────────────▼───────────────────────────────┐
│           CARTE EXPÉRIENCE (µC TBD)                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Alimentation propre]                              │
│                                                     │
│  [IMU propre]                                       │
│                                                     │
│  [Expérience] ←── Contrôle selon séquence          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Conformité aux règles des séquenceurs (Référence)

### Règles appliquées
- ✅ **SEQ1**: Aucune liaison électrique entre séquenceur et autres systèmes (optocoupleur)
- ✅ **SEQ2**: Autonomie minimale 1 heure requise
- ✅ **SEQ3**: Puissance suffisante pour déclencher mécanisme séparation
- ✅ **SEQ4**: Signalisation claire (2 LEDs + état via IHM)
- ✅ **SEQ5**: Fenêtrage temporel [T1, T2] pour sécurité
- ⏳ **SEQ6**: Formule temps déclenchement (à déterminer)
- ⏳ **SEQ7**: Plans séquenceur et câblage (à réaliser dans KiCad)

---

## Prochaines étapes

1. [ ] Schématique KiCad de la carte séquenceur
2. [ ] Routage PCB (contrainte: Ø < 9.8 cm, format rond)
3. [ ] Sélection composants (optocoupleur, connecteurs)
4. [ ] Programmation Teensy 4.1
5. [ ] Tests de validation
6. [ ] Définition carte expérience
7. [ ] Intégration système complet

---

*Projet: Fusée Expérimentale - Électronique Séquenceur*  
*Date: Février 2026*
