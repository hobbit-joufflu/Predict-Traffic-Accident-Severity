import numpy as np
import pandas as pd

def harmonize_safety_equipment(df):

    # Nouveau format
    secu_cols = ['secu1', 'secu2', 'secu3']
    if all(col in df.columns for col in secu_cols):

        df[secu_cols] = df[secu_cols].fillna(8).astype(int)
        
        df['ceinture'] = df[secu_cols].isin([1]).any(axis=1).astype(int)
        df['casque'] = df[secu_cols].isin([2]).any(axis=1).astype(int)
        df['airbag'] = df[secu_cols].isin([5, 7]).any(axis=1).astype(int)
        # 3=Enfant, 1=Ceinture, 2=Casque, 5/7=Airbag
        df['safety'] = df[secu_cols].isin([1, 2, 3, 5, 7]).any(axis=1).astype(int)

    # Format ancien 
    elif 'secu' in df.columns:
        # On convertit en numérique, les erreurs deviennent NaN, puis 88 (non renseigné)
        s = pd.to_numeric(df['secu'], errors='coerce').fillna(0).astype(int)
        
        # Premier chiffre:équipement (1=Ceinture, 2=Casque)
        # Second chiffre:utilisation (1=Oui, 2=Non, 3=Indéterminé)
        
        df['ceinture'] = ((s//10==1) & (s % 10 == 1)).astype(int)

        df['casque'] = ((s//10==2)&(s% 10 == 1)).astype(int)
        
        df['airbag'] = 0  # Non disponible dans l'ancien format
        
        # On considère "safety_used" si on est équipés (1)
        # pour la ceinture (1), le casque (2) ou le dispositif enfant (3)
        df['safety'] = ((s//10).isin([1, 2, 3]) & (s % 10 == 1)).astype(int)

    return df

def trace_route(df_cleaned2224):
    df_cleaned2224["route_danger"] = ((df_cleaned2224["plan"]>1).astype(int) &
                                     (df_cleaned2224["prof"]>1).astype(int)
                                     )

    df_cleaned2224["int_danger"] = df_cleaned2224["prof"]*(df_cleaned2224["int"].isin([4,6,8])).astype(int)

    return df_cleaned2224

def lum_atm_surf_hr(df_cleaned2224):

    if "hrmn" in df_cleaned2224.columns:
        df_cleaned2224["hr"]=df_cleaned2224["hrmn"].astype(str).str[:2].astype(int)
    df_cleaned2224 = df_cleaned2224.drop(columns="hrmn",errors="ignore")

    df_cleaned2224 = df_cleaned2224.copy()
    df_cleaned2224["time_period"] = pd.cut(df_cleaned2224['hr'], 
                               bins=[-1, 7, 19, 24],
                               labels=['Nuit',"Matin et après-midi",'Soirée'])
    
    # Transformation cyclique de l'heure
    df_cleaned2224['hour_sin'] = np.sin(2 * np.pi * df_cleaned2224['hr'] / 24)
    df_cleaned2224['hour_cos'] = np.cos(2 * np.pi * df_cleaned2224['hr'] / 24)

    # Surface inondée ou boueuse ; brouillard ou 
    df_cleaned2224['surface_critique'] = df_cleaned2224["surf"].isin([4,6]).astype(int)
    df_cleaned2224['meteo_critique'] = df_cleaned2224["atm"].isin([5,6]).astype(int)
    
    df_cleaned2224['anomalie_atm'] = (df_cleaned2224['atm'] == 9).astype(int)
    df_cleaned2224['anomalie_surf'] = (df_cleaned2224['surf'] == 9).astype(int)

    # Pas d'éclairage
    df_cleaned2224['not_lum'] = (df_cleaned2224["lum"].isin([3,4])).astype(int)
 
    return df_cleaned2224

def pietons_trajet(df_cleaned2224):
    df_cleaned2224['trajet_haute_grav'] = (df_cleaned2224['trajet'].isin([3,5])).astype(int)
    
    df_cleaned2224['ped_loc_seul'] = (
    (df_cleaned2224['etatp'] == 1) & 
    (df_cleaned2224['locp'].isin([1, 6]))
    ).astype(int)
    
    df_cleaned2224["seul_avec_animal"] = (
        (df_cleaned2224["actp"]==6) & 
        (df_cleaned2224["etatp"]==1)
    ).astype(int)

    return df_cleaned2224
