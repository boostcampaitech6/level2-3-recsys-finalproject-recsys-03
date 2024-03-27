from pymongo import MongoClient
import pandas as pd

def load_data_from_mongodb():
    client = MongoClient("mongodb://your_mongodb_uri")
    db = client.your_database
    collection = db.your_collection

    data = pd.DataFrame(list(collection.find()))
    # 데이터 전처리 (필요한 경우)
    return data