from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
from loguru import logger

from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
    
# TF-IDF 벡터라이저 초기화
      
def db_connection():
    load_dotenv()
    client = MongoClient(os.environ.get('MONGO_URI_1'))

    # 사용할 데이터베이스 선택
    db = client['playlist_recommendation']
    return db

def fetch_and_prepare_data():
    # 데이터 불러오기
    db = db_connection()
    
    collection = db['Track']
    documents = collection.find({}, {'tags' : 1, 'track_name' : 1, 'artist_name' : 1, 'uri' : 1})
    tags, track_names, artist_names, uris = [], [], [], []
    
    for doc in documents:
        tags.append(" ".join(doc['tags']))
        track_names.append(doc['track_name'])
        artist_names.append(doc['artist_name'])
        uris.append(doc['uri'])

    return tags, track_names, artist_names, uris
    

def recommend_songs(input_tags, N=10):
    tags, track_names, artist_names, uris = fetch_and_prepare_data()
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags)
    
    # 입력 태그 TF-IDF 벡터화
    input_vec = tfidf_vectorizer.transform([input_tags])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix)

    # 가장 유사한 노래 찾기 (상위 N개)
    similar_indices = cosine_sim.argsort()[0][-N-1:-1][::-1]

    # 추천된 노래의 track_name 출력
    recommended_songs = [(track_names[i], artist_names[i], uris[i]) for i in similar_indices]

    return recommended_songs


if __name__ == "__main__":
    # 사용자로부터 input taga 받기
    input_tags = "pop Love life"
    # 입력된 태그로 노래 추천
    logger.info(f"추천된 노래의 Track ID: {recommend_songs(input_tags)}")