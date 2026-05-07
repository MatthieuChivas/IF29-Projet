from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Connection 
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['tweets']

# création d'un index pour permettre d'accélérer les requêtes sur user.id
collection.create_index("user.id")

# Création d'une pipeline pour récupérer les tweets de 10000 utilisateurs
pipeline = [
	{"$sort": {"user.id": 1}},
	{
        "$group": {
            "_id": "$user.id",
            "txt": {"$push": "$text"}
        }
  },
	{"$limit": 10000}
]

result = list(collection.aggregate(pipeline))
