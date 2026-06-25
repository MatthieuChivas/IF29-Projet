import re
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# --- 1. CONNEXION ET EXTRACTION MONGO ---
# =====================================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db["tweets"]
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

# =====================================================================
# --- 2. FEATURE ENGINEERING GLOBAL ---
# =====================================================================
print("\n--- Calcul de l'ensemble des variables ---")

df["taux_retweet"] = (df["nb_retweets"] / df["total_tweets_dataset"]) * 100

def compter_moyenne_elements(liste_de_listes):
    total = 0
    for liste in liste_de_listes:
        if isinstance(liste, list):
            total += len(liste)
    return total / len(liste_de_listes) if liste_de_listes else 0

df["moyenne_hashtags"] = df["tous_les_hashtags"].apply(compter_moyenne_elements)
df["moyenne_urls"] = df["toutes_les_urls"].apply(compter_moyenne_elements)
df["moyenne_mentions"] = df["toutes_les_mentions"].apply(compter_moyenne_elements)
df["nb_sources_distinctes"] = df["sources"].apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
df["reputation_ratio"] = df["followers_count"] / (df["followers_count"] + df["friends_count"] + 1)

def calculer_tweets_par_jour_dataset(row):
    try:
        date_min = pd.to_datetime(row["date_premier_tweet"])
        date_max = pd.to_datetime(row["date_dernier_tweet"])
        duration_days = (date_max - date_min).total_seconds() / 86400
        if duration_days <= 0:
            duration_days = 1.0
        return row["total_tweets_dataset"] / duration_days
    except:
        return 0

df["frequence_publication"] = df.apply(calculer_tweets_par_jour_dataset, axis=1)

def extraire_source_principale(liste_sources):
    if not liste_sources:
        return "Inconnue"
    source_brute = max(set(liste_sources), key=liste_sources.count)
    match = re.search(r">([^<]+)<", str(source_brute))
    return match.group(1) if match else str(source_brute)

df["source_principale"] = df["sources"].apply(extraire_source_principale)

SOURCES_BOTS_AVEREES = ["IFTTT", "Paper.li", "dlvr.it", "Tweepsmap", "BotSlayer", "TweetOldPost", "Hootsuite", "Buffer", "SocialOomph", "Sprout Social", "Tweetbot for iOS"]
toutes_les_features = ["nb_sources_distinctes", "reputation_ratio", "frequence_publication", "taux_retweet", "moyenne_hashtags", "moyenne_urls", "moyenne_mentions"]

# =====================================================================
# --- PREMIÈRE PARTIE : TEST SUR LES PEU ACTIFS (SANS RÉENTRAÎNEMENT) ---
# =====================================================================
print("\n" + "="*60 + "\nPARTIE 1 : TEST DU MODÈLE DES ACTIFS (>5) SUR LES PEU ACTIFS (<=5)\n" + "="*60)

# Labellisation classique (uniquement basée sur la source)
df["Y_source"] = df["source_principale"].apply(lambda src: 1 if any(bot_app in src for bot_app in SOURCES_BOTS_AVEREES) else 0)

# Séparation des populations
df_actifs = df[df["total_tweets_dataset"] > 5].copy()
df_peu_actifs = df[df["total_tweets_dataset"] <= 5].copy()

# Entraînement du Random Forest sur les actifs (>5 tweets) avec split 80/20
X_act = df_actifs[toutes_les_features].fillna(0)
y_act = df_actifs["Y_source"]
X_train_act, X_test_act, y_train_act, y_test_act = train_test_split(
    X_act, y_act, test_size=0.20, random_state=42, stratify=y_act
)

rf_actifs = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_actifs.fit(X_train_act, y_train_act)

# --- LE TEST SOUHAITÉ ---
print(f"Évaluation directe sur les {df_peu_actifs.shape[0]} utilisateurs peu actifs (<=5 tweets) :")
X_pa = df_peu_actifs[toutes_les_features].fillna(0)
y_pa = df_peu_actifs["Y_source"]

# Prédiction avec le modèle existant (rf_actifs)
y_pred_pa = rf_actifs.predict(X_pa)

print("\nMatrice de Confusion (sur les <=5 tweets) :")
print(confusion_matrix(y_pa, y_pred_pa))

print("\nRapport de Classification (sur les <=5 tweets) :")
print(classification_report(y_pa, y_pred_pa, target_names=["Normal (0)", "Bot (1)"], zero_division=0))


# =====================================================================
# --- DEUXIÈME PARTIE : MODÈLE GLOBAL AVEC LA NOUVELLE RÈGLE ---
# =====================================================================
print("\n" + "="*60 + "\nPARTIE 2 : MODÈLE GLOBAL AVEC NOUVELLE RÈGLE DE LABELLISATION\n" + "="*60)
print("Règle : Si <= 5 tweets -> Humain (0). Si > 5 tweets -> On regarde la source.")

def labellisation_nouvelle_regle(row):
    if row["total_tweets_dataset"] <= 5:
        return 0  # Pas bot d'office
    else:
        return 1 if any(bot_app in row["source_principale"] for bot_app in SOURCES_BOTS_AVEREES) else 0

df["Y_nouvelle_regle"] = df.apply(labellisation_nouvelle_regle, axis=1)

print("\n=== RÉPARTITION DES NOUVEAUX LABELS GLOBAUX ===")
print(df["Y_nouvelle_regle"].value_counts().rename(index={0: "Humains/Apps Officielles/Peu Actifs (0)", 1: "Bots Actifs Source (1)"}))

# Entraînement global (80/20) sur TOUS les profils confondus
X_global = df[toutes_les_features].fillna(0)
y_global = df["Y_nouvelle_regle"]

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_global, y_global, test_size=0.20, random_state=42, stratify=y_global
)

rf_global = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_global.fit(X_train_g, y_train_g)

print("\n=== ÉVALUATION DU MODÈLE GLOBAL (NOUVELLE RÈGLE) ===")
y_pred_g = rf_global.predict(X_test_g)

print("\nMatrice de Confusion :")
print(confusion_matrix(y_test_g, y_pred_g))

print("\nRapport de Classification :")
print(classification_report(y_test_g, y_pred_g, target_names=["Normal/Peu Actif (0)", "Bot Actif (1)"], zero_division=0))

print("\n=== CLASSEMENT DE L'IMPORTANCE DES VARIABLES (MODÈLE GLOBAL) ===")
importances = rf_global.feature_importances_
indices = importances.argsort()[::-1]

for i in indices:
    print(f"Indicateur '{toutes_les_features[i]}' : {importances[i]*100:.2f}% d'impact")