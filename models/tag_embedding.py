import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_tag_list(sideinfo_data):
    sideinfo_data['tags'] = sideinfo_data['tags'].apply(eval)
    sideinfo_data['tag_list'] = sideinfo_data['tags'].apply(list)
    tags_set = set()
    for tags in sideinfo_data['tags']:
        tags_set.update(tags)

    tag_list = pd.DataFrame(tags_set, columns=['tag'])
    return tag_list


def tag_embedding(tag_list):

    ## Embedding1 : Emotion Embedding 
    # 항목 : joy, anger, sadness, fear, disgust, surprise, neutral
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

    def classify_emotions(tag):
        results = classifier(tag, top_k=7)
        return {result['label']: result['score'] for result in results}

    embedded_1 = tag_list['tag'].apply(lambda tag: pd.Series(classify_emotions(tag)))
    tag_embedded = pd.concat([tag_list[['tag']], embedded_1], axis=1)

    ## Embedding2 : Text Embedding 
    # 항목 : emb_1 ~ 7
    import os
    os.environ['HF_TOKEN'] = "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"

    from transformers import AutoModel
    from numpy.linalg import norm
    from sklearn.decomposition import PCA
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
    embeddings = []
    for tag in tag_list['tag']:
        tag_embedding = model.encode(tag)
        embeddings.append(tag_embedding)

    # PCA
    pca = PCA(n_components=7)
    embedded_2 = pca.fit_transform(embeddings)
    for i in range(7):  
        tag_embedded[f'emb_{i+1}'] = embedded_2[:, i]
    # 반올림
    tag_embedded = tag_embedded.round(4)
    return tag_embedded


def song_embedding(sideinfo,tag_embedded,tag_type):
    #sideinfo['tags'] = sideinfo['tags'].apply(eval)
    columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                    'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']
    tag_embedded = tag_embedded.set_index('tag')
    def calculate_average(tags, tag_embedded, columns):
        valid_tags = tag_embedded.index.intersection(tags)
        return tag_embedded.loc[valid_tags, columns].mean(axis=0) if valid_tags.any() else pd.Series(0.5, index=columns)

    average_values_list = pd.DataFrame(sideinfo[tag_type].apply(lambda x: calculate_average(x, tag_embedded, columns)))
    sideinfo[average_values_list.columns] = average_values_list
    sideinfo.drop(['duration_ms', 'liveness', 'mode', 'time_signature'], axis=1, inplace=True)

    return sideinfo


def scaling(song_embedded):
    columns_embedded = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7' , 
                'danceability', 'energy', 'key', 'loudness','speechiness', 'acousticness', 'instrumentalness',  'valence', 'tempo',  
                #'duration_ms','liveness','mode','time_signature',
                ]                       
    scaler = MinMaxScaler()
    song_embedded[columns_embedded] = scaler.fit_transform(song_embedded[columns_embedded]).round(4)
    return song_embedded

if __name__ == "__main__":
    '''
    sideinfo_data = pd.read_csv('../data/preprocessed_music3.csv', index_col=0)
    tag_list = create_tag_list(sideinfo_data)
    tag_list.to_csv('../data/tag_list.csv')
    tag_embedded = tag_embedding(tag_list) # 3662개 : 30분
    tag_embedded.to_csv('../data/tag_embedded.csv')
    tag_embedded = pd.read_csv('../data/tag_embedded_hyperpersonalized.csv', index_col=0)
    song_embedded = song_embedding(sideinfo_data,tag_embedded)
    song_embedded = scaling(song_embedded)
    song_embedded.to_csv('../data/song_embedded_hyperpersonalized.csv')

    #sideinfo에서 genre 태그만 남기기
    sideinfo_data = pd.read_csv('../data/preprocessed_music3.csv', index_col=0)
    sideinfo_data['tags'] = sideinfo_data['tags'].apply(eval)
    tag_genre_list = pd.read_csv('../data/tag_genre_list.csv')
    genre_tags_list = tag_genre_list['Genre Tags'].tolist()
    sideinfo_data['tags'] = sideinfo_data['tags'].apply(lambda x: [tag for tag in x if tag in genre_tags_list])
    sideinfo_data.to_csv('../data/preprocessed_music3_genre.csv')
    tag_embedded = pd.read_csv('../data/tag_embedded_hyperpersonalized.csv', index_col=0)
    song_embedded = song_embedding(sideinfo_data,tag_embedded)
    song_embedded = scaling(song_embedded)
    song_embedded.to_csv('../data/song_embedded_personalized.csv')
    '''

    # sideinfo 에 interacion exist 추가 (자동화 필요)
    sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    idx_threshold = 5638
    sideinfo['interaction_exist'] = [1 if idx < idx_threshold else 0 for idx in range(len(sideinfo))]
    sideinfo.to_csv('../data/preprocessed_music5.csv')

    # Tags를 tag와 genre 합치기
    sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    sideinfo['tags'] = sideinfo['tags'].apply(eval)
    sideinfo['genres'] = sideinfo['genres'].apply(eval)
    sideinfo['tags'] = sideinfo.apply(lambda row: row['tags'] + row['genres'], axis=1)
    sideinfo.to_csv('../data/preprocessed_music5.csv')

    # Tag embedding 생성
    sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    tag_list = create_tag_list(sideinfo)
    #tag_list.to_csv('../data/tag_list.csv')
    tag_embedded = tag_embedding(tag_list) # 3662개 : 30분
    tag_embedded.to_csv('../data/tag_embedded.csv')

    # Song embedding 생성
    tag_embedded = pd.read_csv('../data/tag_embedded.csv', index_col=0)
    sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    sideinfo['tags'] = sideinfo['tags'].apply(eval)
    song_embedded = song_embedding(sideinfo,tag_embedded,'tags')
    song_embedded = scaling(song_embedded)
    song_embedded.to_csv('../data/song_embedded_hyperpersonalized.csv')

    # Song embedding 생성
    tag_embedded = pd.read_csv('../data/tag_embedded.csv', index_col=0)
    sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    sideinfo['genres'] = sideinfo['genres'].apply(eval)
    song_embedded = song_embedding(sideinfo,tag_embedded,'genres')
    song_embedded = scaling(song_embedded)
    song_embedded.to_csv('../data/song_embedded_personalized.csv')
