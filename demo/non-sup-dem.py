import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Configuration de la page Streamlit
st.set_page_config(page_title="Unsupervised Clustering Dashboard", layout="wide")

st.title("🤖 Analyse de Comportement Twitter - Modèle Non Supervisé")
st.markdown("Dashboard interactif de démonstration de clustering (Gaussian Mixture Model & PCA)")

# ==========================================
# Chargement des données (Cache Parquet d'abord)
# ==========================================
@st.cache_data
def charger_donnees_non_sup():
    # Liste de chemins potentiels pour le cache Parquet
    paths_to_try = [
        "user_features_sample_2.parquet",
        "../user_features_sample_2.parquet",
        "cache_features/user_features_sample_2.parquet",
        "../cache_features/user_features_sample_2.parquet",
        os.path.join(os.path.dirname(__file__), "../user_features_sample_2.parquet"),
        os.path.join(os.path.dirname(__file__), "user_features_sample_2.parquet")
    ]
    
    df = None
    source_utilisee = ""
    for p in paths_to_try:
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                source_utilisee = f"Cache Parquet ({p})"
                break
            except Exception as e:
                pass
                
    if df is None:
        # Fallback connexion MongoDB
        from pymongo import MongoClient
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            db = client["test"]
            collection = db["tweets"]
            
            pipeline = [
                {"$sample": {"size": 25000}},  # Taille raisonnable pour la démo
                {"$match": {"user.id": {"$exists": True, "$ne": None}}},
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
                        },
                        "nb_mentions": {
                            "$sum": {
                                "$size": {"$ifNull": ["$entities.user_mentions", []]}
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
                        "sources": {"$addToSet": "$source"},
                        "first_tweet": {"$min": "$created_at"},
                        "last_tweet": {"$max": "$created_at"},
                        "followers_count": {"$first": "$user.followers_count"},
                        "friends_count": {"$first": "$user.friends_count"},
                        "statuses_count": {"$first": "$user.statuses_count"},
                        "user_created_at": {"$first": "$user.created_at"}
                    }
                },
                {"$match": {"total_tweets_dataset": {"$gt": 5}}}
            ]
            result = list(collection.aggregate(pipeline))
            df = pd.DataFrame(result)
            source_utilisee = "MongoDB (Fallback)"
        except Exception as ex:
            st.error(f"Erreur lors du chargement : cache Parquet introuvable et échec de connexion à MongoDB.")
            st.stop()
            
    if df is None or df.empty:
        st.error("Aucune donnée disponible.")
        st.stop()

    # Feature Engineering
    df["taux_rt"] = df["nb_retweets"] / df["total_tweets_dataset"]
    df["avg_mentions"] = df["nb_mentions"] / df["total_tweets_dataset"]
    df["avg_hashtags"] = df["nb_hashtags"] / df["total_tweets_dataset"]
    df["avg_urls"] = df["nb_urls"] / df["total_tweets_dataset"]
    df["intensite"] = df["avg_mentions"] + df["avg_hashtags"] + df["avg_urls"]
    df["nb_sources"] = df["sources"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray, set)) else 1)
    df["reputation_ratio"] = df["followers_count"] / (df["followers_count"] + df["friends_count"] + 1)
    
    # Fréquence de publication
    df["first_tweet"] = pd.to_datetime(df["first_tweet"])
    df["last_tweet"] = pd.to_datetime(df["last_tweet"])
    df["observation_days"] = (df["last_tweet"] - df["first_tweet"]).dt.total_seconds() / 86400
    df["observation_days"] = df["observation_days"].clip(lower=1)
    df["tweet_frequency"] = df["total_tweets_dataset"] / df["observation_days"]

    return df, source_utilisee

with st.spinner("Chargement des données..."):
    df, source_info = charger_donnees_non_sup()

st.success(f"Données chargées depuis : **{source_info}**")

# ==========================================
# Modélisation GMM
# ==========================================
features = [
    "taux_rt",
    "avg_mentions",
    "avg_hashtags",
    "avg_urls",
    "nb_sources",
    "reputation_ratio",
    "tweet_frequency"
]

# Nettoyage des NaN sur les features ML
df_ml = df.dropna(subset=features).copy()

if len(df_ml) > 10000:
    # On échantillonne un peu pour que l'app Streamlit reste fluide
    df_sample = df_ml.sample(10000, random_state=42).copy()
else:
    df_sample = df_ml.copy()

X = df_sample[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle GMM à 3 clusters
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
df_sample["cluster"] = gmm.fit_predict(X_scaled)
proba = gmm.predict_proba(X_scaled)
df_sample["cluster_proba"] = proba.max(axis=1)
df_sample["log_likelihood"] = gmm.score_samples(X_scaled)

# ==========================================
# Affichage des Statistiques Globale
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("Utilisateurs analysés", len(df_sample))
col2.metric("Nombre de Clusters", 3)
col3.metric("Moyenne Log-Likelihood", f"{df_sample['log_likelihood'].mean():.2f}")

# ==========================================
# Tableau récapitulatif des clusters
# ==========================================
st.subheader("Profil Moyen des Clusters")

summary = df_sample.groupby("cluster")[features].mean()
summary["nb_utilisateurs"] = df_sample["cluster"].value_counts()
st.dataframe(summary.style.format("{:.3f}").background_gradient(cmap="Blues"))

# ==========================================
# Visualisation PCA 2D
# ==========================================
st.subheader("Visualisation 2D des Clusters (Projection PCA)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_sample["pca_1"] = X_pca[:, 0]
df_sample["pca_2"] = X_pca[:, 1]

# Limiter le nombre de points sur le plot pour les performances (max 2000 points)
plot_df = df_sample.sample(min(2000, len(df_sample)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
colors = {0: "#4E79A7", 1: "#F28E2B", 2: "#E15759"}
labels = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}

for cluster_id in range(3):
    sub = plot_df[plot_df["cluster"] == cluster_id]
    ax.scatter(sub["pca_1"], sub["pca_2"], c=colors[cluster_id], label=labels[cluster_id], alpha=0.7, edgecolors="none", s=25)

ax.set_title("Projection PCA des profils utilisateurs")
ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend()
st.pyplot(fig)

# ==========================================
# Inspecteur d'utilisateur
# ==========================================
st.subheader("Inspecter un Utilisateur")

# Initialisation de st.session_state pour l'index utilisateur
if "selected_user_idx" not in st.session_state:
    st.session_state["selected_user_idx"] = 0

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🎲 Sélectionner un utilisateur au hasard"):
        st.session_state["selected_user_idx"] = int(np.random.randint(0, len(df_sample)))
        st.rerun()

with col_btn2:
    if st.button("⚠️ Sélectionner une anomalie au hasard (Log-Likelihood bas)"):
        threshold = df_sample["log_likelihood"].quantile(0.05)
        anomalies = df_sample[df_sample["log_likelihood"] < threshold]
        if not anomalies.empty:
            random_anomaly = anomalies.sample(1)
            # Retrouver l'index de position de cette anomalie
            pos = df_sample.index.get_loc(random_anomaly.index[0])
            st.session_state["selected_user_idx"] = int(pos)
            st.rerun()

user_idx = st.slider("Index de l'utilisateur", 0, len(df_sample) - 1, value=st.session_state["selected_user_idx"])
st.session_state["selected_user_idx"] = user_idx

selected_user = df_sample.iloc[user_idx]

u_col1, u_col2 = st.columns(2)

with u_col1:
    st.write("### Identifiants & Attribution")
    st.write(f"**ID Utilisateur :** `{selected_user['_id']}`")
    st.write(f"**Cluster assigné :** Cluster {selected_user['cluster']}")
    st.write(f"**Confiance (probabilité) :** {selected_user['cluster_proba']*100:.2f}%")
    st.write(f"**Score de vraisemblance (Log-Likelihood) :** {selected_user['log_likelihood']:.2f}")
    if selected_user["log_likelihood"] < df_sample["log_likelihood"].quantile(0.05):
        st.warning("⚠️ Cet utilisateur présente un comportement atypique (potentielle anomalie)")

with u_col2:
    st.write("### Indicateurs de l'utilisateur")
    user_features = pd.DataFrame({
        "Variable": features,
        "Valeur": [selected_user[f] for f in features],
        "Moyenne Globale": [df_sample[f].mean() for f in features]
    })
    st.dataframe(user_features.style.format({"Valeur": "{:.4f}", "Moyenne Globale": "{:.4f}"}))
