import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ==========================================
# 1. Connection à la BDD / Extraction / Chargement
# ==========================================
parquet_path = "cache_features/user_features_all.parquet"

if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    print("Features chargées depuis le cache Parquet.")
else:
    print("Cache absent : Connexion à MongoDB pour extraction...")
    try:
        client = MongoClient("mongodb://localhost:27017/") 
        db = client["projet"] 
        collection = db["MesElements"] 
        print("Connexion OK à MongoDB") 
        
        pipeline = [
            {
                "$match": {
                    "user.id": {"$exists": True, "$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$user.id",
                    # Nombre de tweets dans le dataset
                    "total_tweets_dataset": {"$sum": 1},
                    # Nombre de retweets
                    "nb_retweets": {
                        "$sum": {
                            "$cond": [
                                {"$ifNull": ["$retweeted_status", False]},
                                1,
                                0
                            ]
                        }
                    },
                    # Nombre total de mentions
                    "nb_mentions": {
                        "$sum": {
                            "$size": {
                                "$ifNull": ["$entities.user_mentions", []]
                            }
                        }
                    },
                    # Nombre total de hashtags
                    "nb_hashtags": {
                        "$sum": {
                            "$size": {
                                "$ifNull": ["$entities.hashtags", []]
                            }
                        }
                    },
                    # Nombre total d'URLs
                    "nb_urls": {
                        "$sum": {
                            "$size": {
                                "$ifNull": ["$entities.urls", []]
                            }
                        }
                    },
                    # Sources utilisées
                    "sources": {"$addToSet": "$source"},
                    # Dates des tweets
                    "first_tweet": {"$min": "$created_at"},
                    "last_tweet": {"$max": "$created_at"},
                    # Infos utilisateur
                    "followers_count": {"$first": "$user.followers_count"},
                    "friends_count": {"$first": "$user.friends_count"},
                    "statuses_count": {"$first": "$user.statuses_count"},
                    "user_created_at": {"$first": "$user.created_at"}
                }
            }
            # Supprimé le filtre des utilisateurs actifs (> 5 tweets)
        ]
        
        result = list(collection.aggregate(pipeline, allowDiskUse=True))
        df = pd.DataFrame(result)
        
        # Sauvegarde en Parquet
        os.makedirs("cache_features", exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f"Fichier Parquet sauvegardé : {parquet_path}")
        print(f"Nombre d'utilisateurs stockés : {len(df)}")
    except Exception as e:
        print(f"Erreur lors de la connexion à MongoDB ou de l'extraction : {e}")
        raise e

# ==========================================
# 2. Feature Engineering / Travail sur les variables
# ==========================================
print("Feature engineering...")

# Pour éviter les avertissements pandas, spécifier le format de date mixed/UTC si nécessaire
df["first_tweet"] = pd.to_datetime(df["first_tweet"], errors='coerce')
df["last_tweet"] = pd.to_datetime(df["last_tweet"], errors='coerce')

# Calcul Taux de Retweet :
df["taux_rt"] = (
    df["nb_retweets"] /
    df["total_tweets_dataset"]
)

# Calcul Intensité :
df["avg_mentions"] = (
    df["nb_mentions"] /
    df["total_tweets_dataset"]
)

df["avg_hashtags"] = (
    df["nb_hashtags"] /
    df["total_tweets_dataset"]
)

df["avg_urls"] = (
    df["nb_urls"] /
    df["total_tweets_dataset"]
)

df["intensite"] = (
    df["avg_mentions"] +
    df["avg_hashtags"] +
    df["avg_urls"]
)

# Nombre de sources différentes
df["nb_sources"] = df["sources"].apply(len)

# Ratio de réputation
df["reputation_ratio"] = (
    df["followers_count"] /
    df["friends_count"].replace(0, np.nan)
)

# Fréquence de publication:
df["observation_days"] = (
    (df["last_tweet"] - df["first_tweet"])
    .dt.total_seconds()
    / 86400
)

# Éviter les divisions par 0
df["observation_days"] = df["observation_days"].clip(lower=1)

df["tweet_frequency"] = (
    df["total_tweets_dataset"] /
    df["observation_days"]
)

# ==========================================
# 3. Machine Learning Non Supervisé (GMM)
# ==========================================
features = [
    "taux_rt",
    "avg_mentions",
    "avg_hashtags",
    "avg_urls",
    "nb_sources",
    "reputation_ratio",
    "tweet_frequency"
    # Note : On conserve les mêmes variables d'apprentissage
]

df_ml = df.copy()
df_ml = df_ml.dropna(subset=features)

print(f"Données filtrées pour le Machine Learning : {len(df_ml)} lignes.")

X = df_ml[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Entraînement du modèle GMM (3 clusters)...")
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42
)

df_ml["cluster"] = gmm.fit_predict(X_scaled)

# Probabilité d'appartenance au cluster choisi
proba = gmm.predict_proba(X_scaled)
df_ml["cluster_proba"] = proba.max(axis=1)

# Score d'anomalie : plus c'est bas, plus c'est atypique
df_ml["log_likelihood"] = gmm.score_samples(X_scaled)

# Sauvegarder le modèle et le scaler
joblib.dump(gmm, "gmm_model_all.joblib")
joblib.dump(scaler, "scaler_all.joblib")
print("Modèle GMM et Scaler sauvegardés.")



# ==========================================
# 4. Statistiques des Clusters
# ==========================================
summary = df_ml.groupby("cluster")[features].mean()
counts = df_ml["cluster"].value_counts()

print("\nNombre d'utilisateurs par cluster :")
print(counts)

print("\nMoyenne des variables par cluster :")
print(summary)

# ==========================================
# 5. Visualisations & Sauvegarde des Figures
# ==========================================
print("Calcul de la PCA et génération des graphiques...")

# Trouver le dossier visuel local (non-sup/visuel)
script_dir = os.path.dirname(os.path.abspath(__file__))
visuel_dir = os.path.join(script_dir, "visuel")
os.makedirs(visuel_dir, exist_ok=True)

# 5.1 Enregistrement du tableau des moyennes sous forme d'image
def save_summary_table(df_summary, counts, filepath, title):
    df_table = df_summary.copy()
    df_table.insert(0, "nb_users", counts)
    df_table = df_table.round(4).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, len(df_table) * 0.7 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df_table.values, 
        colLabels=df_table.columns, 
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4E79A7')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('#FFFFFF')
            if col == 0:
                cell.set_text_props(weight='bold')
                
    plt.title(title, fontsize=14, pad=15, weight='bold')
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

table_img_path = os.path.join(visuel_dir, "moyenne_variables_clusters_all.png")
save_summary_table(summary, counts, table_img_path, "Moyenne des variables par cluster (Tous utilisateurs)")
print(f"Tableau des moyennes sauvegardé sous '{table_img_path}'.")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_ml["pca_1"] = X_pca[:, 0]
df_ml["pca_2"] = X_pca[:, 1]

# 5.2 Graphique de projection PCA 2D
plt.figure(figsize=(10, 6))
colors = {0: "#4E79A7", 1: "#F28E2B", 2: "#E15759"}
labels = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}

# Sous-échantillon pour affichage si trop grand (2000 points max)
plot_df = df_ml.sample(min(2000, len(df_ml)), random_state=42) if len(df_ml) > 2000 else df_ml

for cluster_id in range(3):
    sub = plot_df[plot_df["cluster"] == cluster_id]
    plt.scatter(sub["pca_1"], sub["pca_2"], c=[colors[cluster_id]], label=labels[cluster_id], alpha=0.7, edgecolors="none", s=25)

plt.title("Projection PCA des profils utilisateurs (Tous utilisateurs)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
pca_img_path = os.path.join(visuel_dir, "pca_clusters_all.png")
plt.savefig(pca_img_path)
plt.close()
print(f"Graphique PCA sauvegardé sous '{pca_img_path}'.")

# 5.3 Graphique Réputation vs Fréquence de Publication
plt.figure(figsize=(10, 6))
plt.scatter(
    df_ml["reputation_ratio"],
    df_ml["tweet_frequency"],
    c=df_ml["cluster"],
    cmap="viridis",
    alpha=0.6
)
plt.xlabel("Ratio de réputation")
plt.ylabel("Fréquence de publication")
plt.title("Ratio de réputation vs fréquence de publication (Tous utilisateurs)")
plt.colorbar(label="Cluster")
rep_img_path = os.path.join(visuel_dir, "reputation_vs_frequency_all.png")
plt.savefig(rep_img_path)
plt.close()
print(f"Graphique réputation vs fréquence sauvegardé sous '{rep_img_path}'.")
