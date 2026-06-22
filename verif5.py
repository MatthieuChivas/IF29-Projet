from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["projet"]
    collection = db["MesElements"]
    print("Connexion OK à MongoDB")

    # Pipeline 1 : Compter le nombre total d'utilisateurs uniques
    pipeline_total = [
        {
            "$match": {
                "user.id": {"$exists": True, "$ne": None}
            }
        },
        {
            "$group": {
                "_id": "$user.id"
            }
        },
        {
            "$count": "total_users"
        }
    ]

    # Pipeline 2 : Compter les utilisateurs uniques ayant plus de 5 tweets
    pipeline_gt5 = [
        {
            "$match": {
                "user.id": {"$exists": True, "$ne": None}
            }
        },
        {
            "$group": {
                "_id": "$user.id",
                "total_tweets": {"$sum": 1}
            }
        },
        {
            "$match": {
                "total_tweets": {"$gt": 5}
            }
        },
        {
            "$count": "total_users_gt5"
        }
    ]

    print("Calcul en cours...")
    result_total = list(collection.aggregate(pipeline_total))
    result_gt5 = list(collection.aggregate(pipeline_gt5))

    total_all = result_total[0]["total_users"] if result_total else 0
    total_gt5 = result_gt5[0]["total_users_gt5"] if result_gt5 else 0

    print(f"\n--- RÉSULTATS ---")
    print(f"Nombre total d'utilisateurs uniques dans la base : {total_all}")
    print(f"Nombre d'utilisateurs uniques avec plus de 5 tweets : {total_gt5}")

except Exception as e:
    print(f"Une erreur est survenue : {e}")
