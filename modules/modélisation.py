# %% [markdown]
# # Modélisation

# %% [markdown]
# ### 1. prédiction de grav grâce à : heure, age, mois, atm, lumière, surface (random forest)

# %%
!pip install scikit-learn

# %%
# On définit la cible (Y) et les variables explicatives (X)
# On utilise les variables que tu as créées : heure_seule, age, et les dummies
features = ['heure_seule', 'age', 'mois'] + [c for c in df_final.columns if 'atm_' in c or 'surf_' in c or 'lum_' in c]
X = df_final[features]
y = df_final['grav']

# %%
from sklearn.model_selection import train_test_split


# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Taille de l'entraînement : {X_train.shape[0]} lignes")

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
# Initialisation du modèle
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Entraînement
rf_model.fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report, confusion_matrix

# %%
y_pred = rf_model.predict(X_test)

# Affichage du rapport détaillé (Precision, Recall, F1-score)
print(classification_report(y_test, y_pred))

# Matrice de confusion pour voir où le modèle se trompe
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion')
plt.show()

# %% [markdown]
# L'accuracy est de 45%, ce qui n'est pas génial. 
# Commen on l'a vu dans la partie Description (distribution de la gravité), il y a plus d'accidents non graves que graves, ce qui justifie pourquoi notre modèle a tendance à se tromper pour tout ce qui n'est pas "indemne".

# %%
# Récupération de l'importance des features
importances = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importances.head(15), x='importance', y='feature')
plt.title("Top 15 des variables les plus importantes pour prédire la gravité")
plt.show()

# %% [markdown]
# ### 2. Amélioration du modèle précédent - random forest 'balanced'

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
features = ['heure_seule', 'age', 'mois'] + [c for c in df_final.columns if 'atm_' in c or 'surf_' in c or 'lum_' in c]

X = df_final[features]
y = df_final['grav']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)


# %%
y_pred = model_rf.predict(X_test)
print("\n--- Nouveau Rapport de Classification (Équilibré) ---")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion : Modèle avec Poids Équilibrés')
plt.xlabel('Prédiction (Modèle)')
plt.ylabel('Réalité (Données)')
plt.show()

# %% [markdown]
# On voit que l'accuracy a baissé, mais il prédit mieux les accidents plus graves.

# %% [markdown]
# ### 3. Autre amélioration - deux classes de gravité au lieu de quatre (random forest)

# %%
df_binaire = df_final.copy()
df_binaire = df_binaire[df_binaire['grav'] > 0]
mapping = {1.0: 0, 4.0: 0, 2.0: 1, 3.0: 1}
df_binaire['grav_bin'] = df_binaire['grav'].map(mapping)

# %%
features = ['heure_seule', 'age', 'mois'] + [c for c in df_binaire.columns if 'atm_' in c or 'surf_' in c or 'lum_' in c]
X = df_binaire[features]
y = df_binaire['grav_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_bin = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model_bin.fit(X_train, y_train)

# %%

y_pred = model_bin.predict(X_test)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Matrice de Confusion Binaire')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# Comme on pouvait s'y attendre, les résultats sont meilleurs, mais on a perdu beaucoup de précision.

# %% [markdown]
# ### 4. Nouveau modèle, utilisation de presque toutes les variables (KNN)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %%
df_sample = df_final.sample(n=min(20000, len(df_final)), random_state=42)

# %%
drop_cols = ['Num_Acc', 'grav', 'hrmn', 'an_nais'] # car nous avons heure_seule et age dans df_final

# %%
# On convertit les colonnes qui devraient être des nombres
cols_numeriques = ['lat', 'long', 'larrout', 'lartpc', 'nbv']

for col in cols_numeriques:
    # On remplace la virgule par un point
    df_sample[col] = df_sample[col].astype(str).str.replace(',', '.')
    # On transforme en float (les erreurs deviennent NaN)
    df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')

# On remplit les éventuels vides créés par la conversion (par la moyenne)
df_sample[cols_numeriques] = df_sample[cols_numeriques].fillna(df_sample[cols_numeriques].mean())

# %%
X = df_sample.drop(columns=drop_cols)
y = df_sample['grav']

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

# %%
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# %%
X = X.fillna(X.mode().iloc[0])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# %%
y_pred = knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# %%
y_pred = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Indemne', 'Tué', 'Hospit.', 'Léger'],
            yticklabels=['Indemne', 'Tué', 'Hospit.', 'Léger'])
plt.xlabel('Prédictions')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion')
plt.show()

# %%
from sklearn.inspection import permutation_importance

# %%
results = permutation_importance(knn, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1)

import pandas as pd
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': results.importances_mean
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Top 15 des paramètres les plus importants (KNN)')
plt.xlabel('Baisse de précision si la colonne est supprimée')
plt.show()

# %% [markdown]
# Ce modèle n'est pas très précis; il ne prédit pas bien les tués, et à tendance à prédire les blessés légers comme étant indemnes.
# On voit que l'âge reste un facteur déterminant, ainsi que l'équipement de sécurité, mais la météo ne semble pas être le facteur le plus important.