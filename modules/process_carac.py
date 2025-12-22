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