#from tag_embedding import tag_ranking_load_data, tag_ranking
import pandas as pd
import numpy as np
sideinfo_data = pd.read_csv('./data/preprocessed_new_music.csv', index_col=0)
tag_embedded = pd.read_csv('./data/tag_embedded.csv', index_col=0)
song_embedded = pd.read_csv('./data/song_embedded.csv', index_col=0)

song_embedded = song_embedded[['track_name','anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                   'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']]


import json

login_user_data = {
  "user_id": 120322,
  "track_name": [
    'vampire',
    'EASY',
    'bad idea right?']
    }
selected_data = login_user_data['track_name']


selected_song = pd.DataFrame(song_embedded[song_embedded['track_name'].isin(selected_data)].iloc[:,1:].mean()).T

embedding_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                   'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def find_similar_songs(target_song_emotions, all_songs_df):
    similarity_scores = []

    # 모든 노래에 대해 코사인 유사도 계산
    for index, row in all_songs_df.iterrows():
        current_song_emotions = row[embedding_columns].values
        similarity = cosine_similarity(target_song_emotions, current_song_emotions)
        similarity_scores.append((row['track_name'], similarity))

    sorted_similar_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return sorted_similar_songs


similar_songs_list = find_similar_songs(selected_song, song_embedded)

# 결과 출력 (상위 10곡만 출력)
print("Songs similar to 'vampire' (Top 10):")
for song, similarity in similar_songs_list[:10]:
    print(f"{song}: {similarity:.4f}")



