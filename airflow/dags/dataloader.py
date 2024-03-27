import boto3
import pandas as pd
import json
from pymongo import MongoClient

def load_data_from_mongodb():
    client = MongoClient("mongodb://your_mongodb_uri")
    db = client.your_database
    collection = db.your_collection

    data = pd.DataFrame(list(collection.find()))
    # 데이터 전처리 (필요한 경우)
    return data

def get_new_interaction_track():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='prod/WebBeta/MongoDB')
    secret = json.loads(response['SecretString'])
    mongodb_uri = secret['mongodb_uri']
    
    client = MongoClient(mongodb_uri)
    db = client['playlist_recommendation']
    collection = db['User']
    
    data = collection.find({})
    new_track_uris = set()
    for user in data:
        for track_uri in user['new_track']:
            new_track_uris.add(track_uri)    
            
    return list(new_track_uris)