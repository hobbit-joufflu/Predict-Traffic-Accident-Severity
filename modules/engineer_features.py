import numpy as np
import pandas as pd

def trace_route(df_cleaned2224):
    df_cleaned2224["route_danger"] = ((df_cleaned2224["plan"]>1).astype(int) &
                                     (df_cleaned2224["prof"]>1).astype(int)
                                     )

    df_cleaned2224["int_danger"] = df_cleaned2224["prof"]*(df_cleaned2224["int"].isin([4,6,8])).astype(int)

    return df_cleaned2224

def lum_atm_surf_hr(df_cleaned2224):

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

    df_cleaned2224['ped_danger_loc'] = (
    (df_cleaned2224['catu'] == 3) & 
    (df_cleaned2224['locp'].isin([1, 6]))
).astype(int)
    
    df_cleaned2224['ped_loc_seul'] = (
    (df_cleaned2224['etatp'] == 1) & 
    (df_cleaned2224['locp'].isin([1, 6]))
).astype(int)
    return df_cleaned2224
