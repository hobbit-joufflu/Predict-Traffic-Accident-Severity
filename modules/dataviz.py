import matplotlib.pyplot as plt
import seaborn as sns

def plot_grav_score(df_communes, df_accidents_grv, params={}):
    alpha_pop = params.get("alpha_pop",1)
    alpha_acc = params.get("alpha_acc",0.4)

    plt.figure(figsize=(12, 12))

    plt.scatter(
        df_communes['lon_commune'],
        df_communes['lat_commune'],
        s=df_communes['Population']**1.2,
        c='blue',
        alpha=alpha_pop,       # Transparence modérée
        linewidth=0,     # Pas de bordure pour garder l'aspect "point"
        label='Population'
    )

    # Couche 2 : Accidents, colorés selon leur score de gravité 

    sc = plt.scatter(
        df_accidents_grv['long'],
        df_accidents_grv['lat'],
        s=0.2+3*df_accidents_grv['grav_score'],           # Taille fixe très fine, qui grossit avec la gravité de l'accident
        c=df_accidents_grv["grav_score"],
        cmap="Reds", 
        alpha=alpha_acc,       # Transparence pour voir la densité
        linewidth=0,
        label='Accidents',
        )

    plt.title("Carte d'intensité des accidents vs Population (bleu)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    cbar = plt.colorbar(sc) # Barre de couleur pour lire les scores
    cbar.set_label("score de gravité cumulé")

    # Légende forcée (pour voir les points même s'ils sont petits sur la carte)
    lgnd = plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_alpha(1) 

    plt.show()

def plot_pop_nbacc(df_communes,df_accidents,params={}):
    alpha_pop=params.get("alpha_pop",0.3)
    alpha_acc=params.get("alpha_acc",0.3)

    plt.figure(figsize=(10, 10))

    # Couche 1 : Population

    plt.scatter(
        df_communes['lon_commune'],
        df_communes['lat_commune'],
        s=df_communes['Population'],
        c='blue',
        alpha=alpha_pop,       # Transparence modérée
        linewidth=0,     # Pas de bordure pour garder l'aspect "point"
        label='Population'
    )

    # Couche 2 : Accidents

    plt.scatter(
        df_accidents['long'],
        df_accidents['lat'],
        s=0.5,           # Taille fixe très fine
        c='red', 
        alpha=alpha_acc,       # Transparence pour voir la densité
        linewidth=0,
        label='Accidents'
    )

    plt.title("Carte brute : Population (Bleu) vs Accidents (Rouge)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Légende forcée (pour voir les points même s'ils sont petits sur la carte)
    lgnd = plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_alpha(1) 

    plt.show()

def plot_dic_grav(df_cleaned2224,dic={}):
    plt.figure(figsize=(8,12))

    for i,(col,desc) in enumerate(dic.items(),1):
        custom_order = None
        if col=="hr":
            custom_order = list(range(7,24))+list(range(0,7))
        plt.subplot(len(dic),1,i) # Création d'une grille de len(dic) lignes et une colonne

        sns.barplot(x=df_cleaned2224[col],y=df_cleaned2224["grav_weight"],order=custom_order)

        plt.title(f"Gravite moyenne de l'accident selon : {desc}")
        plt.xlabel(desc,fontsize=10)
        plt.ylabel(f"Gravite ponderee",fontsize=10)

    plt.tight_layout()
    plt.show()