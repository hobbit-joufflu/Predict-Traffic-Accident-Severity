
target_map = {1: 0, 2: 2, 3: 1, 4: 1}
lum_map = {1: 0, 2: 1, 5: 1, 3: 2, 4: 2}

atm_map = {
    1: 'normale', 7: 'normale', 8: 'normale',
    2: 'pluie', 3: 'pluie',
    4: 'difficile', 5: 'difficile', 6: 'difficile',
    9: 'autre', -1: 'autre'
}

catr_map = {
    1: 'rapide', 2: 'rapide',
    3: 'secondaire',
    4: 'urbain', 5: 'urbain', 6: 'urbain', 7: 'urbain', 9: 'urbain'
}

manv_map = {
    1: 'statique_droit', 2: 'statique_droit',
    9: 'mouvement', 10: 'mouvement', 11: 'mouvement', 12: 'mouvement',
    13: 'mouvement', 14: 'mouvement', 15: 'mouvement', 16: 'mouvement',
    17: 'mouvement', 18: 'mouvement',
    3: 'danger', 5: 'danger', 6: 'danger', 19: 'danger', 25: 'danger',
    4: 'parking', 20: 'parking', 21: 'parking', 
    22: 'parking', 23: 'parking', 24: 'parking',
    0: 'autre', -1: 'autre', 26: 'autre'
}

col_map = {
    1: 'frontale', 2: 'arriere', 3: 'cote',
    4: 'multiple', 5: 'multiple', 
    6: 'autre', 7: 'autre', -1: 'autre'
}

surf_map = {
    1: 'normale',
    2: 'mouillee', 3: 'mouillee', 4: 'mouillee',
    5: 'glissante', 6: 'glissante', 7: 'glissante', 8: 'glissante', 
    9: 'autre', -1: 'autre'
}

int_map = {
    1: 'hors',
    2: 'simple', 3: 'simple', 4: 'simple', 5: 'simple',
    6: 'complexe', 7: 'complexe', 8: 'complexe', 9: 'complexe'
}

actp_map = {
    0: 'inconnu', 1: 'deplacement', 2: 'deplacement',
    3: 'traversant', 4: 'traversant', 
    5: 'divers', 6: 'divers',
    9: 'autre', 10: 'autre', 11: 'inconnu'
}

def clean_secu(df):
    if 'secu' in df.columns:
        # Format avant 2019. On crée une nouvelle variable ceinture et une nouvelle variable casque.
        df["secu"]=df["secu"]
        df['secu_ceinture'] = (df['secu'].astype(str).str.zfill(2) == '11').astype(int)
        df['secu_casque'] = (df['secu'].astype(str).str.zfill(2) == '21').astype(int)
    else:
        # Format après 2019. On cherche le code 1 (ceinture) ou 2 (casque) 
        # dans n'importe laquelle des 3 colonnes d'équipement.
        cols = ['secu1', 'secu2', 'secu3']
        df['secu_ceinture'] = df[cols].isin([1]).any(axis=1).astype(int)
        df['secu_casque'] = df[cols].isin([2]).any(axis=1).astype(int)
    return df

# What if both are in

def process_features(df):
    if 'grav' in df.columns:
        df['target'] = df['grav'].map(target_map)

    if 'actp' in df.columns:
        df['actp'] = df['actp'].replace({'A': 10, 'B': 11, -1: 0, '-1': 0}).astype(float)

    if 'lum' in df.columns:
        df['lum'] = df['lum'].map(lum_map).fillna(0)

    mappings = {
        'atm': atm_map, 
        'catr': catr_map, 
        'manv': manv_map,
        'col': col_map,
        'surf': surf_map,
        'int': int_map,
        'actp': actp_map
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna('autre')

    return df