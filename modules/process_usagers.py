import pandas as pd 
import numpy as np
 
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