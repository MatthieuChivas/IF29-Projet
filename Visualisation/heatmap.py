import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def extraire_source_principale(liste_sources):
    if liste_sources is None:
        return "Inconnue"
    if isinstance(liste_sources, float):
        return "Inconnue"
        
    # S'assurer que c'est une liste standard
    if isinstance(liste_sources, (list, np.ndarray, set)):
        liste_sources = list(liste_sources)
    else:
        liste_sources = [str(liste_sources)]
        
    if len(liste_sources) == 0:
        return "Inconnue"
        
    source_brute = max(set(liste_sources), key=liste_sources.count)
    match = re.search(r">([^<]+)<", str(source_brute))
    return match.group(1) if match else str(source_brute)

def charger_donnees_visuel():
    paths_to_try = [
        "user_features_sample_2.parquet",
        "../user_features_sample_2.parquet",
        "cache_features/user_features_sample_2.parquet",
        "../cache_features/user_features_sample_2.parquet",
        os.path.join(os.path.dirname(__file__), "../user_features_sample_2.parquet"),
        os.path.join(os.path.dirname(__file__), "user_features_sample_2.parquet")
    ]
    
    df = None
    for p in paths_to_try:
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                print(f"Chargement réussi depuis {p}")
                break
            except Exception as e:
                pass
                
    if df is None:
        # Fallback connexion MongoDB
        from pymongo import MongoClient
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            db = client["projet"]
            collection = db["MesElements"]
            print("Connexion à MongoDB (projet) pour la heatmap...")
            pipeline = [
                {"$sample": {"size": 30000}},
                {"$match": {"user.id": {"$exists": True, "$ne": None}}},
                {
                    "$group": {
                        "_id": "$user.id",
                        "followers_count": {"$first": "$user.followers_count"},
                        "friends_count": {"$first": "$user.friends_count"},
                        "total_tweets_dataset": {"$sum": 1},
                        "nb_retweets": {
                            "$sum": {
                                "$cond": [
                                    {"$ifNull": ["$retweeted_status", False]},
                                    1,
                                    0
                                ]
                            }
                        },
                        "nb_hashtags": {
                            "$sum": {
                                "$size": {"$ifNull": ["$entities.hashtags", []]}
                            }
                        },
                        "nb_urls": {
                            "$sum": {
                                "$size": {"$ifNull": ["$entities.urls", []]}
                            }
                        },
                        "nb_mentions": {
                            "$sum": {
                                "$size": {"$ifNull": ["$entities.user_mentions", []]}
                            }
                        },
                        "sources": {"$push": "$source"},
                        "first_tweet": {"$min": "$created_at"},
                        "last_tweet": {"$max": "$created_at"}
                    }
                },
                {"$match": {"total_tweets_dataset": {"$gt": 5}}}
            ]
            result = list(collection.aggregate(pipeline))
            df = pd.DataFrame(result)
        except Exception as ex:
            print("Erreur : Impossible de charger les données.")
            exit(1)
            
    # Filtre de base > 5 tweets si pas déjà fait
    if df is not None and not df.empty and "total_tweets_dataset" in df.columns:
        df = df[df["total_tweets_dataset"] > 5].copy()
        
    return df

def main():
    df = charger_donnees_visuel()
    
    # Feature Engineering
    print("Calcul des variables...")
    df["ratio_reputation"] = df["followers_count"] / (df["followers_count"] + df["friends_count"] + 1)
    
    df["source_principale"] = df["sources"].apply(extraire_source_principale)
    # Encodage numérique de la source principale
    df["source_principale_encoded"] = LabelEncoder().fit_transform(df["source_principale"].astype(str))
    
    df["nb_sources"] = df["sources"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray, set)) else 1)
    
    # Fréquence publication
    df["first_tweet"] = pd.to_datetime(df["first_tweet"])
    df["last_tweet"] = pd.to_datetime(df["last_tweet"])
    df["observation_days"] = (df["last_tweet"] - df["first_tweet"]).dt.total_seconds() / 86400
    df["observation_days"] = df["observation_days"].clip(lower=1)
    df["frequence_publication"] = df["total_tweets_dataset"] / df["observation_days"]
    
    # Intensité de diffusion
    # S'assurer que les colonnes de comptage d'entités existent (en fallback ou parquet)
    if "nb_mentions" not in df.columns:
        df["nb_mentions"] = 0
    if "nb_hashtags" not in df.columns:
        df["nb_hashtags"] = 0
    if "nb_urls" not in df.columns:
        df["nb_urls"] = 0
    df["intensite_diffusion"] = (df["nb_mentions"] + df["nb_hashtags"] + df["nb_urls"]) / df["total_tweets_dataset"]
    
    # Taux de retweet
    df["taux_retweet"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100

    # Sélection des 6 variables finales
    var_labels = {
        "ratio_reputation": "Ratio réputation",
        "source_principale_encoded": "Source principale",
        "nb_sources": "Nb de sources",
        "frequence_publication": "Fréquence publication",
        "intensite_diffusion": "Intensité diffusion",
        "taux_retweet": "Taux retweet"
    }
    
    df_corr = df[list(var_labels.keys())].rename(columns=var_labels)
    
    # Calcul de la matrice de corrélation
    corr_matrix = df_corr.corr()
    
    # Masque pour cacher la partie supérieure et la diagonale (coefficients à 1.00)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    
    # Tranchage de la matrice et du masque pour enlever la première ligne et la dernière colonne vides
    # Ligne : de 1 à N-1 (exclut "Ratio réputation")
    # Colonne : de 0 à N-2 (exclut "Taux retweet")
    corr_sliced = corr_matrix.iloc[1:, :-1]
    mask_sliced = mask[1:, :-1]
    
    # Création du plot de la Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tentative d'affichage avec Seaborn pour un rendu optimal
    try:
        import seaborn as sns
        sns.heatmap(corr_sliced, mask=mask_sliced, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": .8})
    except ImportError:
        # Fallback Matplotlib pur
        masked_corr = np.ma.masked_where(mask_sliced, corr_sliced)
        im = ax.imshow(masked_corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Coefficient de Corrélation")
        
        # Annotations textuelles
        for i in range(len(corr_sliced.index)):
            for j in range(len(corr_sliced.columns)):
                if j <= i:  # Afficher uniquement la partie strictement inférieure
                    val = corr_sliced.iloc[i, j]
                    text = ax.text(j, i, f"{val:.3f}",
                                   ha="center", va="center", color="black" if abs(val) < 0.5 else "white")
                               
        ax.set_xticks(np.arange(len(corr_sliced.columns)))
        ax.set_yticks(np.arange(len(corr_sliced.index)))
        ax.set_xticklabels(corr_sliced.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_sliced.index)
        
    ax.set_title("Planche 4 : Heatmap de corrélation des variables finales (> 5 Tweets)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    # Enregistrement
    output_dir = os.path.join(os.path.dirname(__file__), "visuel")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "correlation_heatmap.png")
    
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap de corrélation générée avec succès dans : {output_path}")

if __name__ == "__main__":
    main()
