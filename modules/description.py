# %% [markdown]
# # Description

# %% [markdown]
# #### liens entre la météo et la gravité

# %%
!pip install seaborn matplotlib

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
def calcul_taux_grave_safe(df, prefixe):
    # On cherche les colonnes qui commencent par le préfixe (ex: 'lum_')
    cols = [c for c in df.columns if c.startswith(prefixe)]
    
    if not cols:
        print(f"Aucune colonne trouvée avec le préfixe : {prefixe}")
        return pd.DataFrame(columns=['Condition', 'Taux_Grave_%'])
    
    resultats = []
    for col in cols:
        subset = df[df[col] == 1.0]
        if len(subset) > 0:
            # Calcul du taux d'accidents graves
            taux = (subset['grav'].isin([2, 3]).mean()) * 100
            resultats.append({'Condition': col.replace(prefixe, ''), 'Taux_Grave_%': taux})
    
    if not resultats:
        return pd.DataFrame(columns=['Condition', 'Taux_Grave_%'])
        
    return pd.DataFrame(resultats).sort_values('Taux_Grave_%', ascending=False)

# %%
# Affiche les colonnes pour vérifier les noms exacts
print("Colonnes dispo :", [c for c in df_final.columns if '_' in c][:20])

fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# A. Météo
df_atm = calcul_taux_grave_safe(df_final, 'atm_')
if not df_atm.empty:
    sns.barplot(data=df_atm, x='Taux_Grave_%', y='Condition', ax=axes[0], palette='Reds_r')
axes[0].set_title('Météo vs Gravité (% accidents graves)')

# B. Surface
df_surf = calcul_taux_grave_safe(df_final, 'surf_')
if not df_surf.empty:
    sns.barplot(data=df_surf, x='Taux_Grave_%', y='Condition', ax=axes[1], palette='Blues_r')
axes[1].set_title('Surface vs Gravité (% accidents graves)')

# C. Luminosité
df_lum = calcul_taux_grave_safe(df_final, 'lum_')
if not df_lum.empty:
    sns.barplot(data=df_lum, x='Taux_Grave_%', y='Condition', ax=axes[2], palette='Greys_r')
else:
    axes[2].text(0.5, 0.5, "Données de luminosité non trouvées", ha='center')
axes[2].set_title('Lumière vs Gravité (% accidents graves)')

plt.tight_layout()
plt.show()

# %% [markdown]
# On voit qu'il y a plus d'accidents graves quand les conditions météorologiques sont précaires, mais les écarts ne sont pas énormes.

# %% [markdown]
# #### distribution de la gravité

# %%
mapping_grav = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
df_plot = df_final.copy()
df_plot['grav_label'] = df_plot['grav'].map(mapping_grav)

plt.figure(figsize=(10, 6))
ordre = ["Indemne", "Blessé léger", "Blessé hospitalisé", "Tué"]

sns.countplot(data=df_plot, x='grav_label', order=ordre, palette="viridis")

plt.title("Distribution de la gravité des accidents", fontsize=14)
plt.xlabel("Catégorie de gravité")
plt.ylabel("Nombre d'accidents")
plt.show()

# %% [markdown]
# #### profils temporels (heure, mois)

# %%
# 1. On filtre pour les accidents graves
df_graves = df_final[df_final['grav'].isin([3, 2])].copy()

# 2. On extrait l'heure en gérant les valeurs manquantes (NaN)
# 'errors=coerce' transforme les erreurs en NaN, puis on les supprime
df_graves['heure_seule'] = pd.to_numeric(df_graves['hrmn'].astype(str).str.split(':').str[0], errors='coerce')
df_graves = df_graves.dropna(subset=['heure_seule'])
df_graves['heure_seule'] = df_graves['heure_seule'].astype(int)

# 3. Histogramme
plt.figure(figsize=(12, 6))
plt.hist(df_graves['heure_seule'], bins=range(25), edgecolor='black', color='darkred')
plt.title("Répartition horaire des accidents graves")
plt.xlabel("Heure de la journée")
plt.ylabel("Nombre d'accidents")
plt.xticks(range(0, 24))
plt.show()

# %%
df_final['heure_seule'] = pd.to_numeric(df_final['hrmn'].astype(str).str.split(':').str[0], errors='coerce').fillna(0).astype(int)

# %% [markdown]
# On voit que les accidents graves arrivent en fin d'après-midi et en début de  soirée (sortie de travail, moins de vigilance, alcool), ce qui semble logique.

# %%
# 1. On filtre pour ne garder que les accidents graves 
df_graves_mois = df_final[df_final['grav'].isin([3, 2])].copy()

# 2. On trace l'histogramme
plt.figure(figsize=(12, 6))
# On utilise bins=range(1, 14) pour bien centrer les 12 mois (de 1 à 12)
plt.hist(df_graves_mois['mois'], bins=range(1, 14), edgecolor='black', color='darkblue', alpha=0.7, align='left')

plt.title("Répartition mensuelle des accidents graves (Catégories 3 & 4)", fontsize=14)
plt.xlabel("Mois de l'année", fontsize=12)
plt.ylabel("Nombre d'accidents graves", fontsize=12)

# On remplace les chiffres par les noms des mois pour la clarté
plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'])
plt.grid(axis='y', alpha=0.3)
plt.show()

# %% [markdown]
# Il n'y a pas l'air d'y avoir de saisonalité. Il y a un taux plus été pendant les vacances d'été, sans doute car les conducteurs sont moins vigilants et car ils sont dans des endroits qu'ils connaissent moins. 

# %% [markdown]
# #### analyse des âges

# %%
# Calcul dynamique de l'âge au moment de l'accident
if 'an_nais' in df_final.columns and 'year' in df_final.columns:
    # Calcul : Année de l'accident - Année de naissance
    df_final['age'] = df_final['year'] - df_final['an_nais']
    
    # Nettoyage : On supprime les valeurs aberrantes (ex: erreurs de saisie)
    df_final = df_final[(df_final['age'] >= 0) & (df_final['age'] <= 100)]

# Filtrage des accidents graves 
df_graves_age = df_final[df_final['grav'].isin([3, 2])].copy()

# Tracé de l'histogramme
plt.figure(figsize=(12, 6))
sns.histplot(df_graves_age['age'], bins=range(0, 101, 5), kde=True, color='teal', edgecolor='black')

plt.title(f"Distribution des âges lors des accidents graves", fontsize=14)
plt.xlabel("Âge au moment de l'accident", fontsize=12)
plt.ylabel("Nombre d'accidents graves", fontsize=12)
plt.xticks(range(0, 101, 10))
plt.grid(axis='y', alpha=0.3)
plt.show()

# %% [markdown]
# Les accidents les plus graves concernent les jeunes, qui sont moins vigilants et moins habitués à conduire.
