import numpy as np
import pandas as pd

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
    df['nbv'] = df.groupby('catr')['nbv'].transform(lambda x: x.fillna(x.median()))
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