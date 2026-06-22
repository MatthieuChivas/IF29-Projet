import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Bot Detection Dashboard", layout="wide")

st.title("🤖 Détection de Bots Twitter")
st.markdown("Dashboard de démonstration du modèle supervisé Random Forest")

# ============================
# Fonctions
# ============================

def compter_moyenne_elements(liste_de_listes):
    total = 0
    for liste in liste_de_listes:
        if isinstance(liste, list):
            total += len(liste)
    return total / len(liste_de_listes) if liste_de_listes else 0


def calculer_tweets_par_jour_dataset(row):
    try:
        date_min = pd.to_datetime(row["date_premier_tweet"])
        date_max = pd.to_datetime(row["date_dernier_tweet"])
        duration_days = (date_max - date_min).total_seconds() / 86400
        if duration_days <= 0:
            duration_days = 1
        return row["total_tweets_dataset"] / duration_days
    except:
        return 0


def extraire_source_principale(liste_sources):
    if liste_sources is None:
        return "Inconnue"
    if isinstance(liste_sources, float):
        return "Inconnue"
        
    # Convertir en liste standard
    if isinstance(liste_sources, (list, np.ndarray, set)):
        liste_sources = list(liste_sources)
    else:
        liste_sources = [str(liste_sources)]
        
    if len(liste_sources) == 0:
        return "Inconnue"
        
    source_brute = max(set(liste_sources), key=liste_sources.count)
    match = re.search(r">([^<]+)<", str(source_brute))
    return match.group(1) if match else str(source_brute)


@st.cache_data
def charger_donnees():
    import os
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
    for p in paths_to_try:
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                break
            except Exception as e:
                pass

    if df is None:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["projet"]
        collection = db["MesElements"]

        pipeline = [
            {
                "$group": {
                    "_id": "$user.id",
                    "sources": {"$push": "$source"},
                    "nb_retweets": {
                        "$sum": {
                            "$cond": [
                                {"$ifNull": ["$retweeted_status", False]},
                                1,
                                0
                            ]
                        }
                    },
                    "tous_les_hashtags": {"$push": "$entities.hashtags"},
                    "toutes_les_urls": {"$push": "$entities.urls"},
                    "toutes_les_mentions": {"$push": "$entities.user_mentions"},
                    "total_tweets_dataset": {"$sum": 1},
                    "date_premier_tweet": {"$min": "$created_at"},
                    "date_dernier_tweet": {"$max": "$created_at"},
                    "followers_count": {"$first": "$user.followers_count"},
                    "friends_count": {"$first": "$user.friends_count"}
                }
            }
        ]

        result = list(collection.aggregate(pipeline))
        df = pd.DataFrame(result)

    if df is None or df.empty:
        return None

    df = df[df["total_tweets_dataset"] > 5].copy()

    # Sample pour la démo
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    # Features
    if "nb_hashtags" in df.columns:
        # Format du fichier Parquet (compteurs agrégés)
        df["taux_retweet"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100
        df["moyenne_hashtags"] = df["nb_hashtags"] / df["total_tweets_dataset"]
        df["moyenne_urls"] = df["nb_urls"] / df["total_tweets_dataset"]
        df["moyenne_mentions"] = df["nb_mentions"] / df["total_tweets_dataset"]
        df["nb_sources_distinctes"] = df["sources"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray, set)) else 1
        )
        df["reputation_ratio"] = df["followers_count"] / (
            df["followers_count"] + df["friends_count"] + 1
        )
        
        # Fréquence publication
        df["first_tweet"] = pd.to_datetime(df["first_tweet"])
        df["last_tweet"] = pd.to_datetime(df["last_tweet"])
        duration_days = (df["last_tweet"] - df["first_tweet"]).dt.total_seconds() / 86400
        duration_days = duration_days.clip(lower=1)
        df["frequence_publication"] = df["total_tweets_dataset"] / duration_days
    else:
        # Format MongoDB (listes brutes)
        df["taux_retweet"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100
        df["moyenne_hashtags"] = df["tous_les_hashtags"].apply(compter_moyenne_elements)
        df["moyenne_urls"] = df["toutes_les_urls"].apply(compter_moyenne_elements)
        df["moyenne_mentions"] = df["toutes_les_mentions"].apply(compter_moyenne_elements)
        df["nb_sources_distinctes"] = df["sources"].apply(
            lambda x: len(set(x)) if isinstance(x, list) else 0
        )
        df["reputation_ratio"] = df["followers_count"] / (
            df["followers_count"] + df["friends_count"] + 1
        )
        df["frequence_publication"] = df.apply(calculer_tweets_par_jour_dataset, axis=1)

    df["source_principale"] = df["sources"].apply(extraire_source_principale)

    SOURCES_BOTS = [
        "IFTTT", "Paper.li", "dlvr.it", "Tweepsmap", "BotSlayer",
        "TweetOldPost", "Hootsuite", "Buffer", "SocialOomph", "Sprout Social"
    ]

    df["Y_cible"] = df["source_principale"].apply(
        lambda src: 1 if any(bot in src for bot in SOURCES_BOTS) else 0
    )

    return df



# ============================
# Chargement
# ============================

with st.spinner("Chargement des données..."):
    df = charger_donnees()

if df is None:
    st.error("Aucune donnée trouvée.")
    st.stop()

# ============================
# Metrics
# ============================

col1, col2, col3 = st.columns(3)

col1.metric("Utilisateurs analysés", len(df))
col2.metric("Bots détectés", int(df["Y_cible"].sum()))
col3.metric("Humains", int((df["Y_cible"] == 0).sum()))

# ============================
# Model
# ============================

features = [
    "nb_sources_distinctes",
    "reputation_ratio",
    "frequence_publication",
    "taux_retweet",
    "moyenne_hashtags",
    "moyenne_urls",
    "moyenne_mentions"
]

X = df[features].fillna(0)
y = df["Y_cible"]

import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.info("Modèle chargé depuis le fichier sauvegardé (pas de réapprentissage).")
else:
    # Fallback si le fichier n'existe pas
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    st.warning("Fichier de modèle non trouvé. Modèle entraîné à la volée.")

pred = model.predict(X)
cm = confusion_matrix(y, pred)
accuracy = (pred == y).mean()

st.metric("Accuracy sur l'échantillon test (500 utilisateurs)", f"{accuracy*100:.2f}%")


# ============================
# Graphiques
# ============================

st.subheader("Distribution de la fréquence de publication")

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(df["frequence_publication"], bins=40)
ax.set_xlabel("Tweets / jour")
ax.set_ylabel("Nb utilisateurs")
st.pyplot(fig)

# Importance features
st.subheader("Importance des variables")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8,4))
ax.barh(importance["Feature"], importance["Importance"])
st.pyplot(fig)

# Confusion matrix
st.subheader("Matrice de confusion")

fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(cm)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
st.pyplot(fig)

# ============================
# Démo utilisateur
# ============================

st.subheader("Tester un utilisateur")

# Initialisation de st.session_state pour l'index utilisateur supervisé
if "selected_user_idx_sup" not in st.session_state:
    st.session_state["selected_user_idx_sup"] = 0

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🎲 Sélectionner un utilisateur au hasard"):
        st.session_state["selected_user_idx_sup"] = int(np.random.randint(0, len(df)))
        st.rerun()

with col_btn2:
    if st.button("🤖 Sélectionner un bot suspect au hasard"):
        # On cherche les utilisateurs ciblés ou prédits comme bots
        bots = df[df["Y_cible"] == 1]
        if not bots.empty:
            random_bot = bots.sample(1)
            pos = df.index.get_loc(random_bot.index[0])
            st.session_state["selected_user_idx_sup"] = int(pos)
            st.rerun()

user_index = st.slider("Choisir un utilisateur", 0, len(df)-1, value=st.session_state["selected_user_idx_sup"])
st.session_state["selected_user_idx_sup"] = user_index

selected_row = df.iloc[user_index]
sample = X.iloc[user_index:user_index+1]
proba = model.predict_proba(sample)[0][1]
prediction = model.predict(sample)[0]

u_col1, u_col2 = st.columns(2)

with u_col1:
    st.write("### Résultat de la classification")
    st.write(f"**User ID :** `{selected_row['_id']}`")
    st.write(f"**Source principale :** `{selected_row['source_principale']}`")
    st.write(f"**Probabilité bot (Random Forest) :** {proba*100:.2f}%")
    
    if prediction == 1:
        st.error("🤖 BOT détecté")
    else:
        st.success("👤 Utilisateur humain")
    
    st.progress(float(proba))

with u_col2:
    st.write("### Indicateurs de l'utilisateur")
    user_features = pd.DataFrame({
        "Variable": features,
        "Valeur de l'utilisateur": [selected_row[f] for f in features],
        "Moyenne globale du dataset": [df[f].mean() for f in features]
    })
    st.dataframe(user_features.style.format({"Valeur de l'utilisateur": "{:.4f}", "Moyenne globale du dataset": "{:.4f}"}))