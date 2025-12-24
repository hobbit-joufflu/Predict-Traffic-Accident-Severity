# %% [markdown]
# # Pre-processing 
# 
# 
# 
# 

# %% [markdown]
# ## 1. Importer les données

# %%
import pandas as pd
from IPython.display import display
import polars as pl

# %%
# Téléchargement des dataframes depuis la base de données des BAAC.
urls_y = {
    2022: {
        "carac" : "https://www.data.gouv.fr/api/1/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7",
        "lieux" : "https://www.data.gouv.fr/api/1/datasets/r/a6ef711a-1f03-44cb-921a-0ce8ec975995",
        "usagers" : "https://www.data.gouv.fr/api/1/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8"
    },
    2023 : {
        "carac" : "https://www.data.gouv.fr/api/1/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2",
        "lieux" : "https://www.data.gouv.fr/api/1/datasets/r/8bef19bf-a5e4-46b3-b5f9-a145da4686bc",
        "usagers" : "https://www.data.gouv.fr/api/1/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f"
    },
    2024 : {
        "carac" : "https://www.data.gouv.fr/api/1/datasets/r/83f0fb0e-e0ef-47fe-93dd-9aaee851674a",
        "lieux" : "https://www.data.gouv.fr/api/1/datasets/r/228b3cda-fdfb-4677-bd54-ab2107028d2d",
        "usagers" : "https://www.data.gouv.fr/api/1/datasets/r/f57b1f58-386d-4048-8f78-2ebe435df868"
    }
}

url_05_21 = {
    "carac" : "https://www.data.gouv.fr/api/1/datasets/r/a3cac8bc-4a07-4124-8a08-633a3a91d40b",
    "lieux" : "https://www.data.gouv.fr/api/1/datasets/r/b7f25e45-de32-4801-b0eb-62989f1a7406",
    "usagers" : "https://www.data.gouv.fr/api/1/datasets/r/a64b1b9f-4d56-4b26-ae90-9f40b878e109"
}

# Dictionnaire qui contiendra les dataframes pour chaque année en vue de la concaténation

dfs = {}

yrl = [2022,2023,2024]
names = ["carac","lieux","usagers"]

for name in names:
    list_df = []
    for yr in yrl:
        url = urls_y[yr][name]
        df_yr = pd.read_csv(url,sep=";") # Le séparateur utilisé est ";"
        df_yr.insert(1,"year",yr) # Colonne année pour distinguer les accidents entre années en vue des opérations de fusion de dataframes
        list_df.append(df_yr)

    concatenated_df_type = pd.concat(list_df,ignore_index=True) # On fait fi de l'index
    dfs[name] = concatenated_df_type # On associe le dataframe des trois années au type de dataframe

dfs

# Prend environ 40 secondes à tourner

# %% [markdown]
# ## 2. Créer un dataframe pour 2022-2024 regroupant les donneés "Lieux", "Usagers", "Caractéristiques"

# %%
if "an" in dfs["carac"].columns:
    dfs["carac"].drop(columns=["an"],inplace=True) # Colonne an redondante avec colonne year

# %%
if "Accident_Id" in dfs["carac"].columns:
    dfs["carac"]["Num_Acc"] = dfs["carac"]["Num_Acc"].fillna(dfs["carac"]["Accident_Id"])   # On fusionne Accident_Id et Num_Acc, qui représentent le même indicateur
    dfs["carac"].drop(columns=["Accident_Id"],inplace=True)

col_data = dfs["carac"].pop("Num_Acc")
dfs["carac"].insert(0,"Num_Acc",col_data) # On replace la colonne "Num_Acc" en index 0 des colonnes du dataframe

# %% [markdown]
# #### Analyses préliminaires

# %%
for n,content in dfs.items(): 
    print(n,len(content))

# %%
sum(dfs["carac"]["Num_Acc"].value_counts()>1)

# %% [markdown]
# Dans "carac" il n'y a bien qu'une ligne par accident. 
# Il est normal que le dataframe "usagers" soit le plus long : en effet, pour chaque accident corporel, on peut compter plusieurs victimes. Il est plus surprenant que "lieux" soit plus long que "carac". 
# Observons les doublons :

# %%
dfs["lieux"]["Num_Acc"].value_counts()

# %%
dfs["lieux"][dfs["lieux"]["Num_Acc"]==202300035508]

# %% [markdown]
# "Lieux" indique les différentes voies liées à l'accident lorsqu'il a eu lieu dans une intersection complexe. Il contient par ailleurs différentes informations sur le lieu de l'accident (type de route **catr**, l'état de la surface **surf**, la vitesse maximale autorisée **vma**).
# 
# Les données les plus générales sont les "caractéristiques de l'accident", uniques pour chaque accident. Dans chaque accident, on a des données sur le lieu de l'accident et l'avant-accident qui est différent selon les acteurs impliqués. Enfin, la partie "usagers" comporte des informations sur chacun des usagers impliqués dans l'accident ; c'est le dataset le plus large.

# %% [markdown]
# #### Observons "Lieux"

# %% [markdown]
# Nous supprimons la colonne voie, qui est bruitée, et peu informative. On supprime donc également v1 et v2, qui donnent l'adresse. On supprime de même les colonnes concernant les bornes kilométriques (pr et pr1)
# 
# A contrario, la VMA (vitesse maximale autorisée), la catégorie de route (catr - autoroute, route urbaine...), l'état de la surface (surf - conditions de la route : verglas, pluie, neige), le régime de circulation (circ - bidirectionnel, unidirectionnel,...), le type d'infrastructure (infra - ponts, tunnels ou carrefours), situation de l'accident (situ - où a précisément eu lieu l'accident : sur la chaussée, sur la bande d'arrêt d'urgence,...), sont particulièrement importantes pour notre prédiction.
# 
# Les colonnes plan, prof, nbv, lartpc, larrout, vosp, donnent des informations précises sur la configuration des lieux de l'accident. En particulier, prof (topographie), plan (courbure de la route), larrout (largeur de la route) sont très intéressantes.

# %%
dfs["lieux"].drop(columns=["voie","v1","v2", "pr", "pr1"], inplace=True)
dfs["lieux"]

# %%
dfs["lieux"][dfs["lieux"]["Num_Acc"]==202300000001]

# %% [markdown]
# Retirons maintenant les doublons.

# %%
# Garde uniquement la première occurrence de chaque Num_Acc
dfs["lieux"] = dfs["lieux"].drop_duplicates(subset=["Num_Acc"], keep='first')

# %%
dfs["lieux"][dfs["lieux"]["Num_Acc"]==202300000001]

# %% [markdown]
# #### Observons "Usagers"

# %% [markdown]
# Nous supprimons la colonne id_usager ainsi que id_véhicule et num_véhicule, car nous ne nous attarderons pas sur l'analyse des véhicules et nous avons déjà Num_Acc pour identifier les accidents.

# %%
dfs["usagers"].drop(columns=["id_usager","id_vehicule","num_veh"], inplace=True)
dfs["usagers"]

# %% [markdown]
# #### Observons "Caractéristiques"

# %% [markdown]
# Nous pouvons enlever la colonne "adr" car nous avons déjà la latitude et la longitude. Pour la même raison, nous pouvons retirer "dep" et "com", mais nous les gardons de coté si jamais nous souhaitons faire une analyse en fonction des départements et des communes plus tard.

# %%
df_geo_backup_2224 = dfs["carac"][["Num_Acc", "year", "dep", "com"]].copy()
dfs["carac"].drop(columns=["dep","com","adr"], inplace=True) 
dfs["carac"]

# %% [markdown]
# #### Fusionner 2022, 2023 et 2024

# %% [markdown]
# Créons maintenant un premier dataset, qui fusionne les données de 2022,2023, et 2024 pour les dataframes **usagers** et **carac** :

# %%
KEY = ["year","Num_Acc"]

df_final2224 = dfs["usagers"].merge(dfs["carac"],on=KEY,how="left") # On garde toutes les colonnes usagers

print(df_final2224.shape[0]==dfs["usagers"].shape[0])

# df_final2224 et dfs["usagers"] font bien la même longueur.

# %% [markdown]
# Ajoutons-y les données sur les lieux. 

# %%
df_final2224 = df_final2224.merge(dfs["lieux"],on=KEY,how="left") 


# %%
df_final2224

# %%
sum(df_final2224["grav"].isna())

# %%
df_final2224

# %% [markdown]
# La gravité des blessures de chaque victime est renseignée, ce qui nous évite un travail supplémentaire de preprocessing pour la variable d'intérêt **grav**.
# 
# >Pour rappel, **grav** peut prendre quatre valeurs : 
# >* 1 = Indemne
# >* 2 = Tué
# >* 3 = Blessé hospitalisé
# >* 4 = Blessé léger
#     

# %% [markdown]
# ## 3. Créer un dataframe pour 2005-2021 regroupant les donneés "Lieux", "Usagers", "Caractéristiques"

# %% [markdown]
# #### Créer df0521

# %%
url_05_21.items()

# %% [markdown]
# Nous construisons maintenant le dataset contenant les données de 2005 à 2021 en vue d'une harmonisation

# %%
df0521={}
for type,link in url_05_21.items():
    df0521[type] = pd.read_csv(link,
                               encoding="latin-1",  # Old files, with a different encoding than the recent ones
                               sep=",")
df0521

# Prend environ 2 min 30 à tourner

# %% [markdown]
# #### Harmonisation préliminaire

# %% [markdown]
# On supprime la colonne redondante d'index "Unnamed: 0" (liée au format d'importation) et on renomme les colonnes "num_acc" et "annee" pour être en cohérence avec le dataset df_final2224 :

# %%
for type in ["usagers","carac"]:
    df0521[type]=df0521[type].rename(columns={"annee":"year","num_acc":"Num_Acc"})
    if "Unnamed: 0" in df0521[type].columns:
        df0521[type].drop(columns="Unnamed: 0",inplace=True)

df0521


# %%
df0521["lieux"]=df0521["lieux"].rename(columns={"annee":"year","num_acc":"Num_Acc"})
if "Unnamed: 0" in df0521["lieux"].columns:
    df0521["lieux"].drop(columns="Unnamed: 0",inplace=True)

# %% [markdown]
# #### Supprimer les colonnes inutiles

# %% [markdown]
# Maintenant, nous enlevons les mêmes colonnes que nous avions enlevées dans dfs. 

# %%
df0521["lieux"].drop(columns=["voie","v1","v2", "pr", "pr1", "env1"], inplace=True) #en1 n'est que présente entre 2005 et 2021 et signifie la proximité avec une école
df0521["usagers"].drop(columns=["num_veh", "id_vehicule"], inplace=True)
df0521["carac"].drop(columns=["dep","com","adr", "gps"], inplace=True) #gps n'est pas présent dans les données 2022-2024, et nous pouvons la supprimer car nous avons long et lat
df0521


# %%
dfs["lieux"] = dfs["lieux"].drop_duplicates(subset=["Num_Acc"], keep='first')

# %% [markdown]
# #### Merge

# %%
KEY = ["Num_Acc","year"]
df_final0521 = df0521["usagers"].merge(df0521["carac"],on=KEY,how="left")   # On garde toutes les données de "usagers", qui est plus large que "carac".
df_final0521

# %% [markdown]
# Refaisons le même travail que précedemment sur Lieux.Remarquons que dans le dataset ancien figure le feature "env1" qui signifie la proximité d'une école, nous pouvons retirer ce feature. 

# %%
df_final0521 = df_final0521.merge(df0521["lieux"],on=KEY,how="left") 

# %%
df_final0521

# %% [markdown]
# ## 4. Concaténer les deux dataframes

# %% [markdown]
# Observons les différences dans les noms de colonnes du format des anciens datasets (2005-2021), et des nouveaux datasets (2022-2024).

# %%
cols_recentes = set(df_final2224.columns)
cols_anciennes = set(df_final0521.columns)

unique_recent = cols_recentes-cols_anciennes # Soustraire les sets permet de ne garder que les colonnes uniquement présentes en 2022-2024
unique_ancien = cols_anciennes-cols_recentes # Colonnes uniquement présentes en 2005-2021

print(unique_ancien,unique_recent)
print(cols_anciennes,cols_recentes)

# %% [markdown]
# #### Analyse de la donnée securité (one-hot encoding)

# %% [markdown]
# 
# 
# La colonne "secu" correspond de 2005 à 2021 à un code sur deux caractères : 
# - Le premier concerne l'existence d'un équipement de sécurité : 
#     1 – Ceinture
#     2 – Casque
#     3 – Dispositif enfants
#     4 – Equipement réfléchissant
#     9 – Autre 
# 
# - Le second concerne l'utilisation de cet équipement de sécurité :
#     1 – Oui
#     2 – Non
#     3 – Non déterminable
# 
# Dans les versions plus récentes, il est question de l'existence ET de l'utilisation d'un équipement de sécurité, jusqu'à trois à la fois (secu1,secu2,secu3):
#     -1 – Non renseigné  
#     0 – Aucun équipement  
#     1 – Ceinture  
#     2 – Casque  
#     3 – Dispositif enfants  
#     4 – Gilet réfléchissant  
#     5 – Airbag (2RM/3RM)  
#     6 – Gants (2RM/3RM)  
#     7 – Gants + Airbag (2RM/3RM)  
#     8 – Non déterminable  
#     9 – Autre
# 
# Notons que les lignes de code ci-dessus suggèreent que "secu" n'est pas dans la base de données 2022-2024, mais que "secu1", "secu2", "secu3" dans dans la base de données 2005-2021. En effet, on peut supposer qu'un effort d'harmonisation des bases a été entamé. 

# %%
# regardons si ces colonnes sont toutes vides entre 2005 et 2021 (ce n'est visiblement pas les cas)
print(df_final0521[['secu1', 'secu2', 'secu3']].isna().all())

# %%
# regardons la répartition des valeurs dans ces colonnes
print(df_final0521[['secu1', 'secu2', 'secu3']].apply(pd.Series.value_counts).head(10))

# %%
# On regarde le nombre de valeurs non-nulles par année pour chaque colonne
verif_secu = df_final0521.groupby('year')[['secu', 'secu1', 'secu2', 'secu3']].count()
print(verif_secu)


# %% [markdown]
# On constate qu'il y a eu une réforme en 2019. L'harmonisation portera donc sur les données de 2005 à 2018.

# %% [markdown]
# 
# Pour cela, nous proposons de supprimer les colonnes "secu1", "secu2", et "secu3" (ainsi que "secu" pour l'ancienne base de données), et à la place mettre 11 colonnes (pour "non renseigné", "aucun équipement", "ceinture", "casque", etc...), dans lequel 1 signifie que l'équiment existe et est utilisé, et 0 signifie qu'il n'existe pas ou n'a pas été utilisé. (Nous effectuons un one-hot encoding)

# %%
cols_final = ["secu_-1", "secu_0", "secu_1", "secu_2", "secu_3", "secu_4", "secu_5", "secu_6", "secu_7", "secu_8", "secu_9"]
secu_cols = ["secu1", "secu2", "secu3"]

# %% [markdown]
# Regardons le dataframe de 2022 à 2024

# %%
# Initialiser les colonnes finales à 0
for col in cols_final:
    df_final2224[col] = 0

# Remplir les colonnes finales
for c in secu_cols:
    dummies = pd.get_dummies(df_final2224[c], prefix='secu')
    for col in dummies.columns:
        df_final2224[col] = df_final2224[col] | dummies[col]

# Supprimer les colonnes originales
df_final2224.drop(columns=secu_cols, inplace=True)


# %%
print(df_final2224.columns.tolist())

# %% [markdown]
# On fait exactement la même chose sur l'ancienne base de données, en faisant attention à la différence de traitement sur les périodes 2005-2018 et 2019-2021. 

# %%
for col in cols_final:
    df_final0521[col] = 0

# De 2019 à 2021 (Nouveau format) 
for c in ["secu1", "secu2", "secu3"]:
    if c in df_final0521.columns:
        for val in [1, 2, 3, 4, 9]: # Les codes équipements
            col_target = f"secu_{val}"
            if col_target in df_final0521.columns:
                # On met 1 si la colonne c contient la valeur val
                df_final0521.loc[df_final0521[c] == val, col_target] = 1

# De2005 à 2018 (Ancien format)
if 'secu' in df_final0521.columns:
    # On isole les lignes d'avant 2019
    mask_old = df_final0521['year'] < 2019
    
    # On convertit en float puis int pour gérer les NaN proprement
    # On travaille sur une copie temporaire pour extraire les chiffres
    s_temp = df_final0521.loc[mask_old, 'secu'].fillna(0).astype(int)
    
    for val in [1, 2, 3, 4, 9]:
        col_target = f"secu_{val}"
        if col_target in df_final0521.columns:
            # condition : le premier chiffre est val et le deuxième chiffre est 1
            # Ex: 11 (ceinture oui) -> 11 // 10 == 1 et 11 % 10 == 1
            cond_equip = (s_temp // 10 == val)
            cond_usage = (s_temp % 10 == 1)
            
            df_final0521.loc[mask_old & cond_equip & cond_usage, col_target] = 1

df_final0521.drop(columns=["secu", "secu1", "secu2", "secu3"], errors='ignore', inplace=True)

# %%
print(df_final0521.columns.tolist())

# %% [markdown]
# Vérification

# %%
# On vérifie la moyenne de remplissage de secu_1 (ceinture) par année
# Si ça affiche des pourcentages cohérents partout, c'est bon
print("Taux de présence de la ceinture par année :")
print(df_final0521.groupby('year')['secu_1'].mean())
print(df_final2224.groupby('year')['secu_1'].mean())

# %% [markdown]
# On peut dire que ces chiffres sont cohérents, donc que le one-hot encoding a marché.

# %% [markdown]
# #### re-vérification des colonnes des deux dataframes

# %% [markdown]
# Maintenant, revérifions que les dataframes ancien et récent ont bien les mêmes colonnes.

# %%
cols_recentes = set(df_final2224.columns)
cols_anciennes = set(df_final0521.columns)

unique_recent = cols_recentes-cols_anciennes # Soustraire les sets permet de ne garder que les colonnes uniquement présentes en 2022-2024
unique_ancien = cols_anciennes-cols_recentes # Colonnes uniquement présentes en 2005-2021

print(unique_ancien,unique_recent)

# %% [markdown]
# Maintenant que les deux dataframes ont les même colonnes, faisons d'autres vérifications avant de les concaténer. 

# %%
#Réorganisons les colonnes par ordre alphabétique pour faciliter la comparaison
df_final0521 = df_final0521.reindex(sorted(df_final0521.columns), axis=1)
df_final2224 = df_final2224.reindex(sorted(df_final2224.columns), axis=1)

# %%
print(df_final0521.columns.tolist())
print(df_final2224.columns.tolist())

# %% [markdown]
# #### Regardons le type des données dans les deux dataframes, et harmonisons les

# %%
# Regardons les différents types
for col in df_final0521.columns:
     print(col, df_final0521[col].dtype, df_final2224[col].dtype)


# %% [markdown]
# Voici notre stratégie d'harmonisation:
# - On transforme les objet en str (cela concerne : actp, hrmn, lat et long)
# - Transformer les int64 en float64 (car le float64 prend en compte les NaN)
# - Transformer les bool (donc les données sécu) aussi en float64 (pour avoir le moins de type de données différentes dans le dataframe)
# 

# %%
cols_to_str = ['actp', 'hrmn', 'lat', 'long', 'larrout', 'lartpc', 'nbv']

for col in cols_to_str:
    if col in df_final0521.columns:
        df_final0521[col] = df_final0521[col].astype(str)
    if col in df_final2224.columns:
        df_final2224[col] = df_final2224[col].astype(str)

# %%
num_cols_0521 = df_final0521.select_dtypes(include=['number', 'bool']).columns
num_cols_2224 = df_final2224.select_dtypes(include=['number', 'bool']).columns

for col in num_cols_0521:
    df_final0521[col] = df_final0521[col].astype(float)

for col in num_cols_2224:
    df_final2224[col] = df_final2224[col].astype(float)

# %%
# Vérification finale des types
for col in df_final0521.columns:
    if df_final0521[col].dtype != df_final2224[col].dtype:
        print(col, df_final0521[col].dtype, df_final2224[col].dtype)


# %% [markdown]
# C'est bon, les colonnes sont identiques et dans le même ordre, et les valeurs sont de même type, on peut donc concaténer.

# %%
df_final = pd.concat([df_final0521, df_final2224], ignore_index=True)


# %%
df_final 

# %%
print(df_final.columns)


# %% [markdown]
# On choisit d'abandonner la colonne 'place' qui précise la position du passager dans le véhicule car elle n'est pas utilie pour notre projet.

# %%
df_final.drop(columns=['place'], inplace=True)

# %% [markdown]
# ## 5. One-hot encoding 
# 

# %% [markdown]
# #### agg

# %%
print(f"Nombre de NaN dans agg : {df_final['agg'].isna().sum()}")

# %%
df_final['agg'] = df_final['agg'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_agg = {
    -1.0: "agg_non_renseigne",
    1.0: "agg_hors_agglomeration",
    2.0: "agg_en_agglomeration"
}

agg_dummies = pd.get_dummies(df_final['agg'])
agg_dummies = agg_dummies.reindex(mapping_agg.keys(), axis=1, fill_value=0)
agg_dummies = agg_dummies.rename(columns=mapping_agg)

df_final = pd.concat([df_final, agg_dummies], axis=1)
df_final.drop(columns='agg', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'circ' OK : {'circ' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='circ_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='circ_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### atm

# %% [markdown]
# On les cellules vides par un -1.

# %%
valeurs_valides = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0]

# Tout ce qui n'est pas dans cette liste devient -1
df_final.loc[~df_final['atm'].isin(valeurs_valides), 'atm'] = -1.0

# %%
mapping = {
    -1.0: "atm_Non_renseigné",
     1.0: "atm_Normale",
     2.0: "atm_pluie_légère",
     3.0: "atm_pluie_forte",
     4.0: "atm_neige_grêle",
     5.0: "atm_brouillard_fumée",
     6.0: "atm_vent_fort_tempête",
     7.0: "atm_temps_eblouissant",
     8.0: "atm_temps_couvert",
     9.0: "atm_autre"
}

atm_dummies = pd.get_dummies(df_final['atm'])


atm_dummies = atm_dummies.reindex(columns=mapping.keys(), fill_value=0)

atm_dummies = atm_dummies.rename(columns=mapping).astype(float)

df_final = pd.concat([df_final, atm_dummies], axis=1)
df_final.drop(columns=["atm"], inplace=True)

# %%
# Vérification : On regarde si l'atm_Normale est bien la plus fréquente (généralement > 70%)
print(df_final['atm_Normale'].mean())

# %% [markdown]
# #### actp

# %% [markdown]
# Dans la base de données 2022-2024 apparaissent "-1 - Non renseigné", "A - Monte / Descend du véhicule", "B - Inconnue". 
# Nous devons donc remplacer tous les vide ou NaN datant de la période 2005-2021 par des -1, nous allons aussi remplacer tous les B par des -1, et enfin nous allons remplacer tous les A par des 9 qui correspondent à "autre".  Nous remplaçons aussi "0 - Non renseigné ou sans objet" par "-1".
# Puis nous ferrons le one-hot encoding.

# %%
df_final.loc[df_final['actp'] == 'A', 'actp'] = '9'
df_final.loc[df_final['actp'] == 'B', 'actp'] = '-1'


# %%
ratio = (df_final['actp'].isin([0, '0']).sum()) / df_final['actp'].count()
print (ratio)

# %% [markdown]
# On voit que la porportion de 0 est assez significative, mais nous les remplacerons tout de même par -1, car cela révèle la même information.

# %%
df_final.loc[df_final['actp'] == '0', 'actp'] = '-1'

# %%
print(df_final[df_final['year'] == 2010]['actp'].unique())

# %% [markdown]
# On voit qu'il y a un problème de type.

# %%
# On unifie tous les types de "vides" vers le code '-1'
vides = ['0', '0.0', 'nan']
df_final['actp'] = df_final['actp'].replace(vides, '-1').fillna('-1')

# On unifie les codes de catégories
df_final['actp'] = df_final['actp'].replace({
    '1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '5.0': '5', '6.0': '6', '7.0' : '7', '8.0': '8', '9.0': '9'
})

# %%
print(df_final[df_final['year'] == 2010]['actp'].unique())

# %% [markdown]
# Le problème est réglé, nous pouvons faire de one-hot encoding.

# %%
mapping_actp = {
    '-1': "actp_Non_renseigné",
     '1': "actp_Sens_vehicule",
     '2': "actp_Sens_inverse",
     '3': "actp_Traversant",
     '4': "actp_Masqué",
     '5': "actp_Jouant_courant",
     '6': "actp_Avec_animal",
     '9': "actp_Autre"
}

actp_dummies = pd.get_dummies(df_final['actp'])

actp_dummies = actp_dummies.reindex(mapping_actp.keys(), axis=1, fill_value=0)

actp_dummies = actp_dummies.rename(columns=mapping_actp)

df_final = pd.concat([df_final, actp_dummies], axis=1)

df_final.drop(columns='actp', inplace=True)


# %%
# 1. Vérifier que la colonne originale 'actp' a bien été supprimée
if 'actp' not in df_final.columns:
    print("✅ Colonne 'actp' supprimée avec succès.")

# 2. Vérifier si les colonnes créées contiennent bien des données
cols_actp = [col for col in df_final.columns if col.startswith('actp_')]
print("\nMoyenne de présence par catégorie :")
print(df_final[cols_actp].mean())

# 3. Vérification cruciale : est-ce que les années récentes et anciennes sont unifiées ?
# Si tu vois des chiffres cohérents partout, c'est gagné.
print("\nÉvolution de 'Non renseigné' par année :")
print(df_final.groupby('year')['actp_Non_renseigné'].mean())

# %% [markdown]
# On peut dire que le one-hot encoding a fonctionné, et on remarque que le taux de "non renseigné" (ou "sans objet") diminue.

# %%
# Identifier dynamiquement toutes les colonnes créées pour actp
cols_actp_finales = [col for col in df_final.columns if col.startswith('actp_')]

# Transformer ces colonnes en float64
df_final[cols_actp_finales] = df_final[cols_actp_finales].astype('float64')

# Vérification : on affiche le type de chaque colonne actp_
print("Vérification des types pour actp :")
print(df_final[cols_actp_finales].dtypes)

# %% [markdown]
# #### cartr

# %% [markdown]
# Dans la nouvelle base apparaît le 7 qui signifie routes de métropole urbaine. Sa proportion et basse (cf ligne de code ci-dessous), donc on peut l'intégrer dans le 9 qui signigie "autre". A part cela, l'ancienne et la nouvelle base sont les mêmes.

# %%
ratio_7 = (df_final['catr'].astype(str) == '7.0').mean()
print(ratio_7)

# %%
df_final.loc[df_final['catr'] == 7.0, 'catr'] = 9.0


# %%
mapping_catr = {
    1.0: "catr_Autoroute",
    2.0: "catr_Route_nationale",
    3.0: "catr_Route_departementale",
    4.0: "catr_Voie_communale",
    5.0: "catr_Hors_reseau_public",
    6.0: "catr_Parc_stationnement_public",
    9.0: "catr_Autre"
}
catr_dummies = pd.get_dummies(df_final['catr'])
catr_dummies = catr_dummies.reindex(mapping_catr.keys(), axis=1, fill_value=0)
catr_dummies = catr_dummies.rename(columns=mapping_catr)
df_final = pd.concat([df_final, catr_dummies], axis=1)
df_final.drop(columns='catr', inplace=True)


# %%
# 1. Vérifier que la colonne 'catr' n'existe plus
if 'catr' not in df_final.columns:
    print("✅ Colonne 'catr' supprimée avec succès.")

# 2. Lister les nouvelles colonnes et vérifier qu'elles ne sont pas vides
cols_catr = [col for col in df_final.columns if col.startswith('catr_')]
print(f"\nNombre de colonnes catr créées : {len(cols_catr)}")
print("\nRépartition moyenne par catégorie (doit être > 0) :")
print(df_final[cols_catr].mean())

# %% [markdown]
# #### catu

# %% [markdown]
# Dans l'ancien dataframe, il y a la valeur 4 qui signifie piéton en roller ou trotinette. Comme sa porportion est basse (cf ci-dessous), on peut la merge avec le 3 qui signifie simplement "piéton".

# %%
ratio_4_catu = (df_final['catu'] == 4.0).mean()
print(f"Ratio de la catégorie 4 : {ratio_4_catu}")

# %%
df_final['catu'] = df_final['catu'].replace(4.0, 3.0)

# %%
mapping_catu = {
    1.0: "catu_Conducteur",
    2.0: "catu_Passager",
    3.0: "catu_Pieton",
}
catu_dummies = pd.get_dummies(df_final['catu'])
catu_dummies = catu_dummies.reindex(mapping_catu.keys(), axis=1, fill_value=0)
catu_dummies = catu_dummies.rename(columns=mapping_catu)
df_final = pd.concat([df_final, catu_dummies], axis=1)
df_final.drop(columns='catu', inplace=True)


# %%
# Vérifier que 'catu' n'est plus dans le DataFrame
print(f"La colonne 'catu' existe-t-elle encore ? {'catu' in df_final.columns}")

# Vérifier qu'il n'y a plus aucune trace de la catégorie 4 (doit afficher 0)
# On utilise une liste complète pour les nouvelles colonnes
cols_catu = ['catu_Conducteur', 'catu_Passager', 'catu_Pieton']
print(f"Nombre de valeurs '4.0' restantes : {df_final[cols_catu].sum(axis=1).isin([4, 4.0]).sum()}")

# %%
# Vérification des types pour les nouvelles colonnes
print("\nTypes des colonnes catu :")
print(df_final[cols_catu].dtypes)

# Si elles sont en 'uint8' (défaut de get_dummies), convertissez-les ici :
df_final[cols_catu] = df_final[cols_catu].astype('float64')

# %% [markdown]
# #### circ

# %%
df_final['circ'] = df_final['circ'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_circ = {
    -1.0: "circ_Non_renseigné",
    1.0: "circ_sens_unique",
    2.0: "circ_bidirectionnel",
    3.0: "circ_chaussee_sepraree",
    4.0: "circ_voies-d_affectation_varaible"
}
circ_dummies = pd.get_dummies(df_final['circ'])
circ_dummies = circ_dummies.reindex(mapping_circ.keys(), axis=1, fill_value=0)
circ_dummies = circ_dummies.rename(columns=mapping_circ)
df_final = pd.concat([df_final, circ_dummies], axis=1)
df_final.drop(columns='circ', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'circ' OK : {'circ' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='circ_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='circ_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### col

# %%
df_final['col'] = df_final['col'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_col = {
    -1.0: "col_Non_renseigné",
    1.0: "col_deux_vehicules_frontal",
    2.0: "col_deux_vehicules_arrière",
    3.0: "col_deux_vehicules_coté",
    4.0: "col_trois_vehicules_en_chaines",
    5.0 : "col_trois_vehicules_collisions_multiples",
    6.0: "col_autre_collision",
    7.0: "col_sans_collision"
}
col_dummies = pd.get_dummies(df_final['col'])
col_dummies = col_dummies.reindex(mapping_col.keys(), axis=1, fill_value=0)
col_dummies = col_dummies.rename(columns=mapping_col)
df_final = pd.concat([df_final, col_dummies], axis=1)
df_final.drop(columns='col', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'col' OK : {'col' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='col_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='col_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### etatp

# %%
df_final['etatp'] = df_final['etatp'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_etatp = {
    -1.0: "etatp_non_renseigné",
    1.0: "etatp_seul",
    2.0: "etatp_accompagné",
    3.0: "etatp_en_groupe",
}
etatp_dummies = pd.get_dummies(df_final['etatp'])
etatp_dummies = etatp_dummies.reindex(mapping_etatp.keys(), axis=1, fill_value=0)
etatp_dummies = etatp_dummies.rename(columns=mapping_etatp)
df_final = pd.concat([df_final, etatp_dummies], axis=1)
df_final.drop(columns='etatp', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'etatp' OK : {'etatp' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='etatp_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='etatp_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### infra

# %%
ratio_0_infra = (df_final['infra'] == 0.0).mean()
ratio_8_infra = (df_final['infra'] == 8.0).mean()
ratio_9_infra = (df_final['infra'] == 9.0).mean()
print(f"Ratios infra - 0: {ratio_0_infra}, 8: {ratio_8_infra}, 9: {ratio_9_infra}") 

# %% [markdown]
# Même si 8 (Chantiers) et 9(Autres) ont été introduits dans la nouvelle base de données, on les garde. En effet; les rations de 1 et 9 sont comparables.  

# %%
print(f"Nombre de NaN dans infra : {df_final['infra'].isna().sum()}")

# %%
df_final['infra'] = df_final['infra'].fillna(-1.0)


# %%
mapping_infra = {
    -1.0: "infra_non_renseigné",
    0.0: "infra_aucun",
    1.0: "infra_souterrain_tunnel",
    2.0: "infra_pont_autopont",
    3.0: "infra_bretelle_d_echangeur_raccordement",
    4.0: "infra__voie_ferrée",
    5.0: "infra_carrefour_aménagé",
    6.0: "infra_zone_pietonne",
    7.0: "infra_zone_péage",
    8.0: "infra_chantier",
    9.0: "infra_autres"
}
infra_dummies = pd.get_dummies(df_final['infra'])
infra_dummies = infra_dummies.reindex(mapping_infra.keys(), axis=1, fill_value=0)
infra_dummies = infra_dummies.rename(columns=mapping_infra)
df_final = pd.concat([df_final, infra_dummies], axis=1)
df_final.drop(columns='infra', inplace=True)


# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'infra' OK : {'infra' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='infra_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='infra_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### locp

# %%
ratio_0_locp = (df_final['locp'] == 0.0).mean()
print(f"Ratio locp 0: {ratio_0_locp}")

# %%
ratio_1_locp = (df_final['locp'] == -1.0).mean()
print(f"Ratio locp -1: {ratio_1_locp}")

# %% [markdown]
# Même si -1 et 0 ont été introduits dans la nouvelle base de données, ils représentent une part importantn des données, surtout le 0 qui signifie sans objet, ce qui semple logique si beaucoup d'accidents ne sont pas avec un piéton.

# %%
mapping_locp = {
    -1.0: "locp_Non_renseigne",
    0.0 : "Aucun_equpement",
    1.0: "locp_Sur_chaussee_plus_50m_passage",
    2.0: "locp_Sur_chaussee_moins_50m_passage",
    3.0: "locp_Passage_pieton_sans_signalisation",
    4.0: "locp_Passage_pieton_avec_signalisation",
    5.0: "locp_Sur_trottoir",
    6.0: "locp_Sur_accotement",
    7.0: "locp_Sur_refuge_ou_BAU",
    8.0: "locp_Sur_contre_allee"
}


locp_dummies = pd.get_dummies(df_final['locp'])
locp_dummies = locp_dummies.reindex(mapping_locp.keys(), axis=1, fill_value=0)
locp_dummies = locp_dummies.rename(columns=mapping_locp)

df_final = pd.concat([df_final, locp_dummies], axis=1)
df_final.drop(columns='locp', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'locp' OK : {'locp' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='locp_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='locp_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### plan

# %%
df_final['plan'] = df_final['plan'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_plan = {
    -1.0: "plan_non_renseigné",
    1.0: "plan_partie rectiligne",
    2.0: "plan_courbe_gauche",
    3.0: "plan_courbe_droite",
    4.0: "plan_en_S"
}
plan_dummies = pd.get_dummies(df_final['plan'])
plan_dummies = plan_dummies.reindex(mapping_plan.keys(), axis=1, fill_value=0)
plan_dummies = plan_dummies.rename(columns=mapping_plan)
df_final = pd.concat([df_final, plan_dummies], axis=1)
df_final.drop(columns='plan', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'plan' OK : {'plan' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='plan_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='plan_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### prof

# %%
df_final['prof'] = df_final['prof'].fillna(-1.0).replace(0, -1.0)

# %%
mapping_prof = {
    -1.0: "prof_non_renseigné",
    1.0: "prof_plat",
    2.0: "prof_pente",
    3.0: "prof_sommet_cote",
    4.0: "prof_bas_cote"
}
prof_dummies = pd.get_dummies(df_final['prof'])
prof_dummies = prof_dummies.reindex(mapping_prof.keys(), axis=1, fill_value=0)
prof_dummies = prof_dummies.rename(columns=mapping_prof)
df_final = pd.concat([df_final, prof_dummies], axis=1)
df_final.drop(columns='prof', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'plan' OK : {'plan' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='plan_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='plan_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### situ

# %%
df_final['situ'] = df_final['situ'].fillna(-1.0)

# %%
#ratio des catégories présentent que dans le nouveau dataframe
print(f"Ratio de -1 : {(df_final['situ'] == -1.0).mean()}")
print(f"Ratio de 0  : {(df_final['situ'] == 0.0).mean()}")
print(f"Ratio de 6  : {(df_final['situ'] == 6.0).mean()}")
print(f"Ratio de 8  : {(df_final['situ'] == 8.0).mean()}")

#ratio des carégories communes aux deux dataframes
print(f"Ratio de 1 : {(df_final['situ'] == 1.0).mean()}")
print(f"Ratio de 2 : {(df_final['situ'] == 2.0).mean()}")
print(f"Ratio de 3 : {(df_final['situ'] == 3.0).mean()}")
print(f"Ratio de 4 : {(df_final['situ'] == 4.0).mean()}")   
print(f"Ratio de 5 : {(df_final['situ'] == 5.0).mean()}")


# %% [markdown]
# Les ratios ci-dessus justifient le fait de ne pas supprimer les catégories ayant été introduites plus tard.

# %%
mapping_situ = {
    -1.0: "situ_non_renseigne",
    0.0 : "situ_aucun",
    1.0: "situ_sur_chaussee",
    2.0: "situ_sur_bande_arret_urgence",
    3.0: "situ_sur_accotement",
    4.0: "situ_sur_trottoir",
    5.0: "situ_sur_piste_cyclable",
    6.0: "situ_sur_autre_voie_speciale",
    8.0: "situ_autres" }

situ_dummies = pd.get_dummies(df_final['situ'])
situ_dummies = situ_dummies.reindex(mapping_situ.keys(), axis=1, fill_value=0)
situ_dummies = situ_dummies.rename(columns=mapping_situ)
df_final = pd.concat([df_final, situ_dummies], axis=1)
df_final.drop(columns='situ', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'situ' OK : {'situ' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='situ_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='situ_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### surf

# %%
df_final['surf'] = df_final['surf'].fillna(-1.0).replace(0,-1.0)

# %%
mapping_surf = {
    -1.0: "surf_non_renseigne",
    1.0: "surf_normale",
    2.0: "surf_mouillee",
    3.0: "surf_flaques",
    4.0: "surf_inondee",
    5.0: "surf_enneigee",
    6.0: "surf_boue",
    7.0: "surf_verglacee",
    8.0: "surf_corps_gras_huile",
    9.0: "surf_autre"}

surf_dummies = pd.get_dummies(df_final['surf'])
surf_dummies = surf_dummies.reindex(mapping_surf.keys(), axis=1, fill_value=0)
surf_dummies = surf_dummies.rename(columns=mapping_surf)
df_final = pd.concat([df_final, surf_dummies], axis=1)
df_final.drop(columns='surf', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'surf' OK : {'surf' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='surf_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='surf_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### trajet

# %%
df_final['trajet'] = df_final['trajet'].fillna(-1.0).replace(0,-1.0)

# %%
mapping_trajet = {
    -1.0: "trajet_non_renseigne_ou_autre",
    1.0: "trajet_domicile_travail",
    2.0: "trajet_domicile_ecole",
    3.0: "trajet_courses_achats",
    4.0: "trajet_utilisation_professionnelle",
    5.0: "trajet_promenade_loisirs",
    9.0: "trajet_autre"
}

trajet_dummies = pd.get_dummies(df_final['trajet'])
trajet_dummies = trajet_dummies.reindex(mapping_trajet.keys(), axis=1, fill_value=0)
trajet_dummies = trajet_dummies.rename(columns=mapping_trajet)
df_final = pd.concat([df_final, trajet_dummies], axis=1)
df_final.drop(columns='trajet', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'trajet' OK : {'trajet' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='trajet_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='trajet_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### vosp

# %%
df_final['vosp'] = df_final['vosp'].fillna(-1.0)

# %%
mapping_vosp = {
    -1.0: "vosp_non_renseigne",
    0.0: "vosp_sans_objet",
    1.0: "vosp_piste_cyclable",
    2.0: "vosp_bande_cyclable",
    3.0: "vosp_voie_reservee"
}

vosp_dummies = pd.get_dummies(df_final['vosp'])
vosp_dummies = vosp_dummies.reindex(mapping_vosp.keys(), axis=1, fill_value=0)
vosp_dummies = vosp_dummies.rename(columns=mapping_vosp)
df_final = pd.concat([df_final, vosp_dummies], axis=1)
df_final.drop(columns='vosp', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'vosp' OK : {'vosp' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='vosp_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='vosp_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### lum

# %%
mapping_lum = {
    1.0: "lum_plein_jour",
    2.0: "lum_crépuscule_aube",
    3.0: "lum_nuit_sans_eclairage_public",
    4.0: "lum_nuit_avec_eclairage_public_non_allumé",
    5.0: "lum_nuit_avec_eclairage_public_allumé",
}

lum_dummies = pd.get_dummies(df_final['lum'])
lum_dummies = lum_dummies.reindex(mapping_lum.keys(), axis=1, fill_value=0)
lum_dummies = lum_dummies.rename(columns=mapping_lum)
df_final = pd.concat([df_final, lum_dummies], axis=1)
df_final.drop(columns='lum', inplace=True)

# %%
# 1. Vérifier que l'originale est bien supprimée
print(f"Suppression 'lum' OK : {'lum' not in df_final.columns}")

# 2. Vérifier que les nouvelles colonnes sont bien en bool
print(f"Type bool OK : {df_final.filter(like='lum_').dtypes.unique() == 'bool'}")

# 3. Vérifier qu'il y a bien une seule catégorie activée par ligne (somme = 1)
print(f"Cohérence lignes OK : {df_final.filter(like='lum_').sum(axis=1).mean() == 1.0}")

# %% [markdown]
# #### fin du one-hot encoding

# %% [markdown]
# Tout a subi un one-hot encoding sauf :
# - Num_Acc
# - an_nai
# - gravité (entre 1 et 4)
# - hrmn
# - int
# - jour
# - larrout
# - lartpc
# - lat 
# - long

# %%
print(df_final.columns.tolist())

# %%
df_final.head(10)