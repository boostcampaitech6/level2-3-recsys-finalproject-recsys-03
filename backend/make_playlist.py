from filter import filter_model
import os
import sys

import pandas as pd
from pymongo import MongoClient
from config import config
import time

# 상위 디렉토리(level2-3-recsys-finalproject-recsys-03)에 포함된 모듈과의 연결
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.inference import inference

def find_document_by_uri(uri):
    client = MongoClient(config.db_url)
    db = client['playlist_recommendation']
    users_collection = db['User']
    document = users_collection.find_one({'uri': uri}, {'user_id': 1, 'top_track.track_id': 1})
    
    if document:
        print("Find A User Data")
        data = []
        user_id = document['user_id']
        for track in document['top_track']:
            data.append({'user_id': user_id, 'track_id': track})
        tracks_df = pd.DataFrame(data)
        return tracks_df
    else:
        return None

def make_playlist(question, uri, type, tag_list):
    start = time.time()
    # 유저 시청 이력 수집
    login_user_data = find_document_by_uri(uri)
    # 채팅 입력 -> 적합한 태그 5개 선택

    if type=="chat":
        input_tag = ', '.join(filter_model(question, tag_list))
    elif type=="tag":
        input_tag = question
    print(input_tag)

    middle = time.time()
    song_list = inference(login_user_data, input_tag)
    end = time.time()
    print(f"middle time : {middle - start:.5f} sec")
    print(f"end time : {end - middle:.5f} sec")

    return song_list, input_tag


if __name__ == "__main__":
    uri = "31i3jfgnv7d544wr2ywt47wp4nje"
    df_tags = pd.read_csv('../data/tag_list.csv')
    tags = df_tags.tag
    
    question1 = "운동할 때 듣기 좋은 KPop 노래 추천해줘"
    question2 = ""
    #question3 = "언제나 취해 있어야 한다. 모든 것이 거기에 있다. 그것이 유일한 문제다. 그대의 어깨를 짓누르고, 땅을 향해 그대 몸을 구부러뜨리는 저 시간의 무서운 짐을 느끼지 않으려면, 쉴새없이 취해야 한다.그러나 무엇에? 술에, 시에 혹은 미덕에, 무엇에나 그대 좋을 대로. 아무튼 취하라.그리하여 때때로, 궁전의 섬돌 위에서, 도랑의 푸른 풀 위에서, 그대의 방의 침울한 고독 속에서, 그대 깨어 일어나, 취기가 벌써 줄어들거나 사라지거든, 물어보라, 바람에, 물결에, 별에, 새에, 시계에, 달아나는 모든 것에, 울부짖는 모든 것에, 흘러가는 모든 것에, 노래하는 모든 것에, 말하는 모든 것에, 물어보라, 지금이 몇시인지. 그러면 바람이, 물결이, 별이, 새가, 시계가, 그대에게 대답하리라. 지금은 취할 시간! 시간의 학대받는 노예가 되지 않으려면, 취하라, 끊임없이 취하라! 술에, 시에 혹은 미덕에, 그대 좋을 대로."
    #question4 = "맥주 넘치지 않게 따라줘"
    
    # for question in [question1, question2, question3, question4]:
    print(question1)
    print(make_playlist(question1, uri, tags))
    print("-"*100)
    print(question2)
    print(make_playlist(question1, uri, tags))
    print("-"*100)
    