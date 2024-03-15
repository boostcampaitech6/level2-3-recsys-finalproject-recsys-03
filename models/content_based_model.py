#from tag_embedding import tag_ranking_load_data, tag_ranking
import pandas as pd
import numpy as np
import json

def combining_track(login_user_data,song_embedded,top_k):
    song_embedded = song_embedded[['track_id','anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                    'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']]
    selected_data = login_user_data['track_id'][:top_k]
    selected_song = pd.DataFrame(song_embedded[song_embedded['track_id'].isin(selected_data)].iloc[:,1:].mean()).T
    return selected_song

def combining_tag(input_tags,tag_embedded):
    input_tags_list = [tag.strip() for tag in input_tags.split(',')]
    selected_song = pd.DataFrame(tag_embedded[tag_embedded['tag'].isin(input_tags_list)].iloc[:,1:].mean()).T
    return selected_song

def content_based_model(selected_song,song_embedded,top_k):

    embedding_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                    'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']

    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # 두 벡터 중 하나라도 크기가 0이면 유사도를 0으로 반환
        return dot_product / (norm_vec1 * norm_vec2)

    def find_similar_songs(target_song_emotions, all_songs_df):
        similarity_scores = []

        # 모든 노래에 대해 코사인 유사도 계산
        for index, row in all_songs_df.iterrows():
            current_song_emotions = row[embedding_columns].values
            similarity = cosine_similarity(target_song_emotions, current_song_emotions)
            similarity_scores.append((row['track_id'], similarity))

        sorted_similar_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_similar_songs

    similar_songs_list = find_similar_songs(selected_song, song_embedded)
    top_k_similar_songs = similar_songs_list[:top_k]
    top_k_songs_list = [int(song) for song, similarity in top_k_similar_songs]
    return top_k_songs_list
