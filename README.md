# 🚗 Predict the severity of traffic accidents 
*Projet du cours de Python de M1 de l'ENSAE*

Le but de ce repo est de prédire la gravité d'un accident corporel à partir d'une multitude de caractéristiques sur le lieu, la météo, et les circonstances de l'accident. En particulier, est-ce que des données météorologiques objectives et enrichies permettent une meilleure prédiction de la gravité d'un accident que les données subjectives déjà présentes dans la base ?

La baseline sera la donnée "atm" de la BAAC, une donnée des conditions atmosphérique au moment et au lieu de l'accident.

On importe les données de la BAAC, la base de données des accidents corporels liés à la circulation routière. 

Dans la base de données des bulletins d'analyse des accidents corporels de la circulation (BAAC), la définition suivante est donnée : 

"Un accident corporel (mortel et non mortel) de la circulation routière relevé par les forces de l'ordre : 
- implique au moins une victime ,
- survient sur une voie publique ou privée, ouverte à la circulation publique,
- implique au moins un véhicule.

On se centre sur trois jeux de données en particulier :  
- Caract_[ANNEE] : Décrit les circonstances générales de l'accident.
- Lieux_[ANNEE] : Décrit le lieu principal de l'accident, même si celui-ci s'est déroulé à une intersection.
- Usagers_[ANNEE] : Données sur les usagers impliqués.
On importe les données de la BAAC, la base de données des accidents corporels liés à la circulation routière.