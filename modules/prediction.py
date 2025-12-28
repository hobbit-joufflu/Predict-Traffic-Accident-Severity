import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def get_pipeline(categorical_features,n_jobs=None):
    """ Crée la pipeline (preprocessor pour variables catégorielles + random forest classifier)"""
    n_jobs_actual = n_jobs if n_jobs is not None else -1 # Utilise tous les coeurs du processeur pour paralléliser le calcul des arbres de décision

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder = "passthrough" # Les variables numériques sont déjà traitées, donc elles passent telles quelles
    )

    # Utilisation d'un Random Forest avec poids équilibrés car les classes "Tué" (2) 
    # et "Grave" (3) sont minoritaires par rapport aux "Indemnes" (1) 

    # 1. Pipeline avec Verbose
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced',  # On accorde proportionnellement plus d'importance aux blessés hospitalisés
            max_depth=20,
            n_jobs=n_jobs_actual,      # Utilise tout le CPU
            verbose=1       # Affiche la progression des arbres
        ))
    ], verbose=True) # Affiche le temps par étape (Preprocess vs Train)

    return pipeline

def run_cross_val(pipeline,X_train,y_train,n_jobs=None):
    """ Exécute la cross-validation et affiche les scores """
    n_jobs_actual = n_jobs if n_jobs is not None else -1

    # On utilise StratifiedKFold pour préserver la proportion des classes de gravité
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Début de la cross-validation (cela peut prendre 15 à 20 minutes)")

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=n_jobs_actual) 
    print(f"\nF1-score moyen : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return cv_scores

def evaluate_predictions(pipeline, X_test, y_test, labels):
    """Génère le rapport et la matrice de confusion"""
    y_pred = pipeline.predict(X_test)
    
    print("\nRapport de classification détaillé :")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return cm

def get_feature_importance_df(pipeline, categorical_features, numerical_features):
    """Extrait l'importance des variables pour la prédiction selon la diminution moyenne de leur impureté de Gini dans la forêt aléatoire"""
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = cat_feature_names + numerical_features
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False) #  On crée un dataframe contenant les noms des features et leur ordre décroissant d'importance (en premier, la variable qui réduit
    # le plus l'impureté lors de la création des branches de l'arbre)
    
    return df_imp