import pandas as pd 
import numpy as np

def clean_carac(carac_new):
    """
    Nettoie les variables environnementales du fichier carac_new.
    Imputation par le mode pour les faibles volumes de NaN.
    """
    carac_new = carac_new.copy()
    
    # Liste des variables catégorielles à traiter par le mode
    # lum (4 NaN), int (13 NaN), atm (5 NaN), col (82 NaN)
    vars_to_impute = ['lum', 'int', 'atm', 'col']
    
    for col in vars_to_impute:
        if col in carac_new.columns:
            most_frequent = carac_new[col].mode()[0]
            # Remplacement des NaN et des codes d'erreur (-1)
            carac_new[col] = carac_new[col].replace(-1, most_frequent).fillna(most_frequent)

    if "Accident_Id" in carac_new.columns:
        carac_new["Num_Acc"] = carac_new["Num_Acc"].fillna(carac_new["Accident_Id"])   
        carac_new.drop(columns=["Accident_Id"],inplace=True)

    col_data = carac_new.pop("Num_Acc")
    carac_new.insert(0,"Num_Acc",col_data)

    carac_new=carac_new.drop(columns=["adr"],errors="ignore")
                
    return carac_new



def clean_usagers(usagers_new):
    usagers_new=usagers_new.copy()
    """
    On corrige les valeurs manquantes du jeu de données usagers_new.
    """
    df_obj = usagers_new.select_dtypes(['object'])
    usagers_new[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    usagers_new.replace([-1,"-1"], np.nan, inplace=True)

    # La place dans le véhicule ne concerne que 13 enregistrements. Nous pouvons supprimer ces valeurs manquantes.
    usagers_new = usagers_new.dropna(subset="place")

    # On supprime les usagers en fuite, qui représente 0.1% des données
    if "grav" in usagers_new.columns:
        usagers_new = usagers_new[usagers_new["grav"]!=-1]

    # On transforme actp en numérique

    # Le -1 dans "trajet" correspond à la valeur 0, "non renseigné". De même pour la localisation des piétons. Pour etatp (le piéton était-il seul ou accompagné ?), nous remplaçons les valeurs manquantes par 0. Le -1 dans les variables secu peut être assimilé au 8, "non déterminable".
    cols_pietons = ["trajet","locp","etatp","actp"]
    for col in cols_pietons:
        usagers_new[col] = pd.to_numeric(usagers_new[col],errors="coerce").fillna(0).astype(int)
    for i in range(1,4):
        usagers_new[f"secu{i}"] = pd.to_numeric(usagers_new[f"secu{i}"],errors="coerce").fillna(8).astype(int)

    # Nous calculons l'âge, supprimons l'année de naissance, et remplaçons les données manquantes par la médiane.
    # En effet, il y a un biais de l'âge vers les plus jeunes. On évite de prendre la moyenne.
    if "an_nais" in usagers_new.columns:
        usagers_new["age"] = usagers_new["year"]-usagers_new["an_nais"]
    median_age = usagers_new["age"].median()
    usagers_new["age"]=usagers_new["age"].fillna(median_age)
    usagers_new.drop(columns="an_nais",inplace=True,errors="ignore")

    # On remplace le sexe par la valeur la plus fréquente
    mode_sexe = usagers_new["sexe"].mode()[0]
    usagers_new["sexe"]=usagers_new["sexe"].fillna(mode_sexe)

    return usagers_new



def clean_lieux(df):
    """
    Identifie les données manquantes.
    Impute les variables catégorielles par le mode.
    """
    # Conversion des valeurs non renseignées
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace([-1,"-1"], np.nan, inplace=True)

    # Suppression variable inexploitable (>90% NaN)
    df = df.drop(columns=['lartpc'], errors='ignore')

    # Imputation groupée par type de route (catr)
    df['nbv']=pd.to_numeric(df["nbv"],errors="coerce")
    df['nbv'] = df.groupby('catr')['nbv'].transform(lambda x: x.fillna(min(x.median(),1)))

    # Pour les catr/nbv vides, on remplace par la médiane nbv
    df['nbv'] = df['nbv'].fillna(df['nbv'].median())

    # Imputation groupée par type de route (catr) et nombre de voies (nbv)
    # Niveau 1 : Médiane par (Catégorie de route + Nb de voies)
    df['larrout']=pd.to_numeric(df["larrout"],errors="coerce")
    df["larrout"] = df.groupby(['catr', 'nbv'])["larrout"].transform(lambda x: x.fillna(x.median()))
    # Niveau 2 : Médiane par Catégorie de route uniquement
    df["larrout"] = df.groupby('catr')["larrout"].transform(lambda x: x.fillna(x.median()))
     # Niveau 3 : Médiane globale (sécurité finale)
    df["larrout"] = df["larrout"].fillna(df["larrout"].median())

    # Imputation croisée : surf via atm (météo) du dataset carac
    mask = df['surf'].isna() | (df['surf'].isin([-1, 0]))
    
    df.loc[mask & (df['atm'] == 1), 'surf'] = 1               # Temps sec -> Route sèche
    df.loc[mask & (df['atm'].isin([2, 3, 7, 8])), 'surf'] = 2 # Pluie/brouillard -> Mouillée
    df.loc[mask & (df['atm'] == 4), 'surf'] = 5               # Neige -> Enneigée
    
    # Mode résiduel pour surf
    df['surf'] = df['surf'].fillna(df['surf'].mode()[0])

    # Mode pour les autres variables catégorielles
    cols = ['circ', 'vosp', 'prof', 'plan', 'infra', 'situ']
    for c in cols:
        if c in df.columns:
            m = df[c].mode()[0]
            df[c] = df[c].replace(-1, m).fillna(m)

    return df
