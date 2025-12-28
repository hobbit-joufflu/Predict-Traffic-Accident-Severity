# 🚗 Prédire la gravité des accidents de la route 
*Projet du cours de Python de M1 de l'ENSAE*

Ce projet a pour objectif de prédire la gravité des blessures lors d'accidents de la route en France, en exploitant [les bases de données BAAC (Bulletin d'Analyse des Accidents Corporels)](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024). Les données sont publiées annuellement, et considèrent uniquement les accidents corporels (i.e. les accidents pour lesquels il y a au moins un blessé). 

On se concentre sur quatre jeux de données pour chaque année :  
- **Caract_[ANNEE]** : Décrit les caractéristiques générales de l'accident.
- **Lieux_[ANNEE]** : Décrit le lieu principal de l'accident, même si celui-ci s'est déroulé à une intersection.
- **Usagers_[ANNEE]** : Données sur les usagers impliqués.
- **Vehicules_[ANNEE]** : Données sur les véhicules impliqués dans l'accident.

Nous souhaitons prédire le mieux possible l'état de chaque victime (Indemne, blessée, blessée hospitalisée, tuée) selon les circonstances de l'accident et ses caractéristiques. Nous allons notamment étudier comment l'ajout de nouvelles variables (notamment sur la vitesse) modifient la performance prédictive du modèle de prédiction.

## Problématique & Enjeux
Les bases BAAC peuvent-elles nous permettre de prédire l'état des victimes après un accident de la route de manière fiable ? Comment l'enrichissement récent des bases BAAC transforme-t-il notre capacité à anticiper les accidents les plus graves ?

Le projet compare deux époques : 

1. **Période 2005-2018** : Large historique, mais absence de données sur la vitesse (VMA), et définition différente des blessés.

2. **Période 2019-2024** : Données enrichies permettant une modélisation plus fine de la dynamique des accidents.

Pour dépasser les limites descriptives des variables brutes, nous avons créé différentes variables d'interaction (feature engineering) pour améliorer les performances prédictives du modèle. 

Nous avons choisi un algorithme de forêt aléatoire pour entraîner ce dernier, algorithme particulièrement adapté à notre jeu de données.

## Résultats du modèle (Random Forest)
Le modèle de classification multiclasse (Indemne, Léger, Hospitalisé, Tué) montre une progression nette avec l'ajout des variables cinétiques : 

| Indicateur de performance | Période 2005 - 2018 (Sans la variable de vitesse maximale autorisée) | Période 2019 - 2024 (Avec la variable de vitesse maximale autorisée) |
| --------- | --------- | --------- |
| **Accuracy Globale** | 0.61 |  0.65 |
| **F1-Score (Weighted)** | 0.62 | 0.65 |
| **Recall - Classe "Tué"** | 0.63 | 0.39 |
| **Précision - Classe "Tué"** |0.14 | 0.20 |
| **Recall - "Hospitalisé"** | 0.36 | 0.53 |

Si le modèle historique identifie plus de décès (recall élevé), il est très alarmiste (précision faible). L'intégration de la vitesse maximale autorisée (VMA) dans le modèle récent permet d'affiner la prédiction et de réduire significativement les faux positives, rendant le modèle plus exploitable en pratique.

>N.B. : Il convient de noter que le Random Forest est un algorithme par essence stochastique. Nous avons donc fixé la graine pour avoir systématiquement le même résultat.

## Structure du projet
Le projet est conçu de manière modulaire pour assurer la reproductibilité.
- `code.ipynb` : Notebook principal 
- `modules` : Dossier contenant les différents scripts utilisés au sein du notebook. 
- `modules/process.py` : Nettoyage des données BAAC (Caractéristiques, Lieux, Usagers, Véhicules)
- `modules/process_vma.py` : Codes liés à l'analyse plus précise de la variable VMA, vitesse maximale autorisée
- `modules/dataviz.py` : Génération des graphiques et des cartographies
- `modules/engineer_features` : Calcul des cross-features et feature engineering pour améliorer les performances du modèle
- `modules/prediction.py` : Fonctions d'entraînement du modèle et de visualisation des résultats du modèle

## Installation et dépendances
```bash
git clone https://github.com/hobbit-joufflu/Predict-Traffic-Accident-Severity.git

pip install -r requirements.txt
