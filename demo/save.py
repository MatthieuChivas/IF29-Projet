import re
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# --- 1. CONNEXION ET EXTRACTION MONGO (VISION TOTALE DES VARIABLES) ---
# =====================================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["projet"]
collection = db["MesElements"]
print("Connexion OK à MongoDB")

collection.create_index("user.id")

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
            "friends_count": {"$first": "$user.friends_count"},
            "statuses_count": {"$first": "$user.statuses_count"},
            "user_created_at": {"$first": "$user.created_at"}
        }
    }
]

print("Extraction des données...")
result = list(collection.aggregate(pipeline))
df = pd.DataFrame(result)

if df.empty:
    print("Aucune donnée trouvée.")
    exit()

# FILTRE : Population active complète (> 5 tweets)
df_actifs = df[df["total_tweets_dataset"] > 5].copy()
print(f"Nombre total d'utilisateurs actifs retenus (> 5 tweets) : {df_actifs.shape[0]}")


# =====================================================================
# --- 2. FEATURE ENGINEERING GLOBAL  ---
# =====================================================================
print("\n--- Calcul de l'ensemble des variables ---")

df_actifs["taux_retweet"] = (df_actifs["nb_retweets"] / df_actifs["total_tweets_dataset"]) * 100

def compter_moyenne_elements(liste_de_listes):
    total = 0
    for liste in liste_de_listes:
        if isinstance(liste, list):
            total += len(liste)
    return total / len(liste_de_listes) if liste_de_listes else 0

df_actifs["moyenne_hashtags"] = df_actifs["tous_les_hashtags"].apply(compter_moyenne_elements)
df_actifs["moyenne_urls"] = df_actifs["toutes_les_urls"].apply(compter_moyenne_elements)
df_actifs["moyenne_mentions"] = df_actifs["toutes_les_mentions"].apply(compter_moyenne_elements)

df_actifs["nb_sources_distinctes"] = df_actifs["sources"].apply(lambda x: len(set(x)) if isinstance(x, list) else 0)

df_actifs["reputation_ratio"] = df_actifs["followers_count"] / (df_actifs["followers_count"] + df_actifs["friends_count"] + 1)

def calculer_tweets_par_jour_dataset(row):
    try:
        # Conversion des dates récupérées de Mongo
        date_min = pd.to_datetime(row["date_premier_tweet"])
        date_max = pd.to_datetime(row["date_dernier_tweet"])
        
        # Calcul de la durée en jours (total_seconds / 86400)
        duration_days = (date_max - date_min).total_seconds() / 86400
        
        # Si l'utilisateur a tout posté dans la même seconde, on applique un minimum de 1 jour
        if duration_days <= 0:
            duration_days = 1.0
            
        # Fréquence = nombre de tweets dans le dataset / durée dans le dataset
        return row["total_tweets_dataset"] / duration_days
    except:
        return 0

# Application de la fonction
df_actifs["frequence_publication"] = df_actifs.apply(calculer_tweets_par_jour_dataset, axis=1)


# =====================================================================
# --- 3. LABELLISATION PAR LA SOURCE APP ---
# =====================================================================
print("\n--- Étape 1 : Labellisation exclusive par Source Applicative ---")

def extraire_source_principale(liste_sources):
    if not liste_sources:
        return "Inconnue"
    source_brute = max(set(liste_sources), key=liste_sources.count)
    match = re.search(r">([^<]+)<", str(source_brute))
    return match.group(1) if match else str(source_brute)

df_actifs["source_principale"] = df_actifs["sources"].apply(extraire_source_principale)

# Liste des plateformes d'automatisation / bots
SOURCES_BOTS_AVEREES = ["IFTTT", "Paper.li", "dlvr.it", "Tweepsmap", "BotSlayer", "TweetOldPost", "Hootsuite", "Buffer", "SocialOomph", "Sprout Social", "Tweetbot for iOS"]

# Règle cible (Y) : 1 si outil de bot, 0 sinon
df_actifs["Y_cible"] = df_actifs["source_principale"].apply(
    lambda src: 1 if any(bot_app in src for bot_app in SOURCES_BOTS_AVEREES) else 0
)

print("\n=== RÉPARTITION DES LABELS ===")
print(df_actifs["Y_cible"].value_counts().rename(index={0: "Humains/Apps Officielles (0)", 1: "Bots Source (1)"}))


# =====================================================================
# --- 4. MODÉLISATION SUPERVISÉE  ---
# =====================================================================
print("\n--- Étape 2 : Entraînement global  ---")

# Fusion de toutes les features
toutes_les_features = [
    "nb_sources_distinctes", "reputation_ratio", "frequence_publication", "taux_retweet", "moyenne_hashtags", "moyenne_urls", "moyenne_mentions"             
]

X = df_actifs[toutes_les_features].fillna(0)
y = df_actifs["Y_cible"]

# Split 80% Train / 20% Test (Stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Modèle entraîné
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)


# =====================================================================
# --- 5. ÉVALUATION ET SENS DE DÉCISION (LE MATCH DES FEATURES) ---
# =====================================================================
print("\n=== ÉVALUATION DU MODÈLE GLOBAL ===")
y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de Confusion :")
print(cm)

report = classification_report(y_test, y_pred, target_names=["Normal (0)", "Bot (1)"], output_dict=True)
print("\nRapport de Classification :")
print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Bot (1)"]))

print("\n=== CLASSEMENT DE L'IMPORTANCE DES VARIABLES ===")
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]

feature_importances_dict = {}
for i in indices:
    feat = toutes_les_features[i]
    imp = float(importances[i])
    feature_importances_dict[feat] = imp
    print(f"Indicateur '{feat}' : {imp*100:.2f}% d'impact dans la décision")


# =====================================================================
# --- 6. SAUVEGARDE DE L'APPRENTISSAGE DANS MONGODB ---
# =====================================================================
print("\n--- Étape 3 : Sauvegarde de l'apprentissage dans MongoDB ---")
runs_collection = db["ModelRuns"]

run_document = {
    "date_execution": datetime.now(),
    "modele": {
        "type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced"
        }
    },
    "dataset": {
        "nb_utilisateurs_actifs": int(df_actifs.shape[0]),
        "nb_train": int(X_train.shape[0]),
        "nb_test": int(X_test.shape[0]),
        "ratio_bots": float(y.mean())
    },
    "metrics": {
        "accuracy": float((y_pred == y_test).mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    },
    "feature_importances": feature_importances_dict
}

insert_result = runs_collection.insert_one(run_document)
print(f"Apprentissage enregistré avec succès dans la collection 'ModelRuns'.")
print(f"ID du document enregistré : {insert_result.inserted_id}")

# Sauvegarde du modèle dans un fichier pickle
import pickle
import os
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)
print(f"Modèle sauvegardé dans {model_path}")

