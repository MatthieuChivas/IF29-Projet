import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
        # Fallback connexion MongoDB avec filtre > 5 tweets
        from pymongo import MongoClient
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            db = client["projet"]
            collection = db["MesElements"]
            print("Connexion à MongoDB (projet) pour les données abonnés/abonnements...")
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
                        }
                    }
                },
                {"$match": {"total_tweets_dataset": {"$gt": 5}}}
            ]
            result = list(collection.aggregate(pipeline))
            df = pd.DataFrame(result)
        except Exception as ex:
            print("Erreur : Impossible de charger les données abonnés/abonnements.")
            exit(1)
            
    # S'assurer que les colonnes nécessaires pour les features existent
    if df is not None and not df.empty:
        if "total_tweets_dataset" in df.columns:
            # On applique le filtre > 5 tweets si ce n'est pas déjà fait
            df = df[df["total_tweets_dataset"] > 5].copy()
            
    return df

def charger_donnees_retweets():
    # Tente de requêter MongoDB projet.MesElements avec filtre > 5 tweets
    from pymongo import MongoClient
    df = None
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        db = client["projet"]
        collection = db["MesElements"]
        print("Connexion à MongoDB (projet) pour le taux de retweet (> 5 tweets)...")
        
        # On échantillonne un grand nombre de tweets pour estimer la distribution des utilisateurs actifs
        pipeline = [
            {"$sample": {"size": 300000}},
            {
                "$group": {
                    "_id": "$user.id",
                    "total_tweets_dataset": {"$sum": 1},
                    "nb_retweets": {
                        "$sum": {
                            "$cond": [
                                {"$ifNull": ["$retweeted_status", False]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {"$match": {"total_tweets_dataset": {"$gt": 5}}}
        ]
        result = list(collection.aggregate(pipeline))
        df = pd.DataFrame(result)
        df["taux_rt"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100
        print("Données de retweet récupérées avec succès de MongoDB.")
    except Exception as e:
        print("Erreur de connexion à MongoDB, utilisation du cache Parquet (fallbacks actifs uniquement).")
        # Fallback : charger depuis le cache Parquet
        df_parquet = charger_donnees_visuel()
        if df_parquet is not None:
            df = pd.DataFrame()
            df["total_tweets_dataset"] = df_parquet["total_tweets_dataset"]
            df["taux_rt"] = (df_parquet["nb_retweets"] / df_parquet["total_tweets_dataset"]) * 100
            
    return df

def plot_kde_or_hist(ax, data, title, is_active_filtered=False):
    # Couleur bleu comme dans l'image
    color_hist = "#A5D6F7" if not is_active_filtered else "#80C0FF"
    color_line = "#4090E0" if not is_active_filtered else "#1A80E0"
    
    # Histogramme
    counts, bins, patches = ax.hist(data, bins=20, range=(0, 100), color=color_hist, alpha=0.7, edgecolor="white", density=False)
    
    # Tentative d'ajout d'une ligne KDE (scipy)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, 100, 200)
        # Ajustement d'échelle du KDE par rapport aux effectifs réels
        bin_width = 100 / 20
        ax.plot(x_vals, kde(x_vals) * len(data) * bin_width, color=color_line, lw=2)
    except Exception as e:
        pass

    # Ligne verticale seuil suspect à 80%
    ax.axvline(x=80, color="red", linestyle="--", lw=1.5, label="Seuil Suspect (>80%)")
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Taux de Retweet (%)", fontsize=10)
    ax.set_ylabel("Nombre d'utilisateurs", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left")

def generer_graphe_retweet(df_rt, output_dir):
    if df_rt is None or df_rt.empty:
        print("Impossible de générer le graphique du taux de retweet (données manquantes).")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Planche 1 : Analyse de l'Amplification (Taux de Retweet)", fontsize=15, fontweight="bold")
    
    # Utilisateurs actifs (> 5 tweets)
    plot_kde_or_hist(ax1, df_rt["taux_rt"], "Utilisateurs Actifs (> 5 Tweets)", is_active_filtered=False)
    
    # Utilisateurs très actifs (> 20 tweets)
    df_tres_actifs = df_rt[df_rt["total_tweets_dataset"] > 20]
    plot_kde_or_hist(ax2, df_tres_actifs["taux_rt"], "Utilisateurs Très Actifs (> 20 Tweets)", is_active_filtered=True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "distribution_taux_retweet.png")
    plt.savefig(output_path, dpi=150)
    print(f"Graphique de retweet généré avec succès dans : {output_path}")

def generer_graphe_retweet_vs_hashtag(df, output_dir):
    # Feature Engineering
    if "taux_rt" not in df.columns:
        df["taux_rt"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100
    if "avg_hashtags" not in df.columns:
        df["avg_hashtags"] = df["nb_hashtags"] / df["total_tweets_dataset"]
        
    df_clean = df.dropna(subset=["taux_rt", "avg_hashtags"]).copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Planche 2 : Relation Taux de Retweet vs Hashtags Moyens (> 5 Tweets)", fontsize=15, fontweight="bold")
    
    # 1. Scatterplot
    ax1.scatter(df_clean["taux_rt"], df_clean["avg_hashtags"], alpha=0.4, color="#F28E2B", edgecolors="none", s=15)
    ax1.set_xlabel("Taux de Retweet (%)", fontsize=11)
    ax1.set_ylabel("Nombre moyen de hashtags par tweet", fontsize=11)
    ax1.set_title("1. Scatterplot (Taux de Retweet vs Hashtags)", fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # 2. Densité (Hexbin)
    hb = ax2.hexbin(
        df_clean["taux_rt"], 
        df_clean["avg_hashtags"], 
        gridsize=40, 
        cmap="Oranges", 
        norm=colors.LogNorm(), 
        mincnt=1
    )
    ax2.set_xlabel("Taux de Retweet (%)", fontsize=11)
    ax2.set_ylabel("Nombre moyen de hashtags par tweet", fontsize=11)
    ax2.set_title("2. Densité de population (Concentration)", fontsize=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    cb = fig.colorbar(hb, ax=ax2, label="Nombre d'utilisateurs par zone")
    cb.set_label("Nombre d'utilisateurs par zone", fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "retweet_vs_hashtag.png")
    plt.savefig(output_path, dpi=150)
    print(f"Graphique Retweet vs Hashtag généré avec succès dans : {output_path}")

def main():
    # Dossier visuel de sortie
    output_dir = os.path.join(os.path.dirname(__file__), "visuel")
    os.makedirs(output_dir, exist_ok=True)

    # Chargement global des données (uniquement > 5 tweets)
    df = charger_donnees_visuel()

    # 1. GENERER LE GRAPHE ABONNE / ABONNEMENT
    df_clean = df.dropna(subset=["followers_count", "friends_count"]).copy()
    df_clean["followers_log"] = df_clean["followers_count"].clip(lower=0.1)
    df_clean["friends_log"] = df_clean["friends_count"].clip(lower=0.1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Planche 3 : Analyse des Abonnés vs Abonnements (> 5 Tweets)", fontsize=15, fontweight="bold")
    
    # Scatterplot (gauche)
    ax1.scatter(df_clean["friends_log"], df_clean["followers_log"], alpha=0.4, color="#1DA1F2", edgecolors="none", s=15)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Nombre d'abonnements (Friends) - Échelle Log", fontsize=11)
    ax1.set_ylabel("Nombre d'abonnés (Followers) - Échelle Log", fontsize=11)
    ax1.set_title("1. Scatterplot individuel (Abonnés vs Abonnements)", fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Ligne 1:1
    lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, color='red', linestyle='--', alpha=0.7, label='Ratio 1:1')
    ax1.legend()
    
    # Densité (droite)
    hb = ax2.hexbin(
        df_clean["friends_log"], 
        df_clean["followers_log"], 
        gridsize=50, 
        cmap="Blues", 
        norm=colors.LogNorm(), 
        xscale='log', 
        yscale='log', 
        mincnt=1
    )
    ax2.set_xlabel("Nombre d'abonnements (Friends) - Échelle Log", fontsize=11)
    ax2.set_ylabel("Nombre d'abonnés (Followers) - Échelle Log", fontsize=11)
    ax2.set_title("2. Densité de population (Concentration)", fontsize=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    # Ligne 1:1
    lims2 = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims2, lims2, color='red', linestyle='--', alpha=0.7, label='Ratio 1:1')
    ax2.legend()
    
    cb = fig.colorbar(hb, ax=ax2, label="Nombre d'utilisateurs par zone")
    cb.set_label("Nombre d'utilisateurs par zone", fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "abonne_abonnement.png")
    plt.savefig(output_path, dpi=150)
    print(f"Graphique abonnés/abonnements généré avec succès dans : {output_path}")

    # 2. GENERER LE GRAPHE TAUX DE RETWEET (uniquement > 5 tweets)
    df_rt = charger_donnees_retweets()
    generer_graphe_retweet(df_rt, output_dir)

    # 3. GENERER LE GRAPHE RETWEET VS HASHTAG (uniquement > 5 tweets)
    generer_graphe_retweet_vs_hashtag(df, output_dir)

if __name__ == "__main__":
    main()
