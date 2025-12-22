import numpy as np

def analyze_vma_outliers(df):
    """
    Analyse le contexte des VMA > 130 avant correction
    """
    outliers = df[df['vma'] > 130].copy()
    # On regarde la répartition par type de route (catr)
    # 1: Autoroute, 2: Nationale, 3: Départementale, 4: Communale
    summary = outliers.groupby(['vma', 'catr']).agg(
        nb_cas=('Num_Acc', 'count'),
        communes=('Code INSEE', lambda x: list(x.unique())[:3]) # Top 3 villes
    ).reset_index()
    return summary

def clean_and_impute(df):
    """
    Corrige les fautes de frappe (supprime les 0 en trop)
    Supprime les autres valeurs hors plage légale ([10,130])
    Remplace les valeurs manquantes par la médiane pour ce type de route.
    """
    df = df.copy()
    df=df[df["vma"]!=140]
    df.loc[df["vma"]>200,"vma"]=df.loc[df["vma"]>200,"vma"].astype(str).str[:2].astype(int)
    
    df.loc[(df['vma'] < 10) | (df['vma'] > 130),"vma"]=np.nan
    
    df["vma"]=df.groupby('catr')["vma"].transform(lambda x:x.fillna(x.median()))

    print("Nettoyage effectué")
    return df
    
def get_commune_stats(df, df_densite):
    """
    Agrège les données de VMA par commune et fusionne avec la densité.
    """
    # Agrégation par commune (Code INSEE)
    stats = df.groupby('Code INSEE').agg(
        vma_moyenne=('vma', 'mean'),
        gravite_moyenne=('grav_weight', 'mean'),
        nb_accidents=('Num_Acc', 'count')
    ).reset_index()
    
    # Jointure avec la densité
    res = stats.merge(df_densite, on="Code INSEE", how='inner')
    return res