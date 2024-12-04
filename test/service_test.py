import pandas as pd
import os
import sys
from sklearn.metrics import recall_score
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from backend.filter import filter_model
from models.inference import inference

def get_recommendation(question, login_user_data, tag_list):
    print(question)
    tags = filter_model(question, tag_list)
    input_tag = ', '.join(tags)
    song_list = inference(login_user_data, input_tag)
    return song_list


#prepare playlists
playlists = pd.read_csv('playlists.csv')
playlists['track_list'] = playlists['track_id'].apply(lambda x: x.split(','))
available = playlists[playlists['track_list'].apply(len) >= 40]  #20: (278, 3)  #30: (195, 4)
available = available.assign(ans=available['track_list'].apply(lambda x: x[:20]))
available = available.assign(input=available['track_list'].apply(lambda x: x[20:]))

#get recommendation
df_tags = pd.read_csv('../data/tag_list.csv')
tags = df_tags.tag
result=[]


for playlist in available.iterrows():
    question = playlist[1]['playlist_name']
    login_user_data = playlist[1]['input']

    #프롬프트 엔지니어링쪽에서 오류가 나서 일단 빼버림
    if "Top Songs -" in playlist[1]['playlist_name']:
        continue
    data = []
    user_id = -1
    for track in playlist[1]['input']:
        data.append({'user_id': user_id, 'track_id': track})
    tracks_df = pd.DataFrame(data)

    song_list = get_recommendation(question, tracks_df, tags)

    #song id
    id_list = [song['id'] for song in song_list]
    result.append({'chat': question, 'top_track': playlist[1]['input'], 'ans': playlist[1]['ans'],'recommended': id_list})

#compare
result_df = pd.DataFrame(result)
result_df.to_csv('service_test.csv')