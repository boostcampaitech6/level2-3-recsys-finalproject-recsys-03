import pandas as pd
import ast
import numpy as np



def create_tag_list(sideinfo_data):
    def str_to_dict(s):
        try:
            return ast.literal_eval(s)
        except ValueError:
            return {}
        
    sideinfo_data['tags_dict'] = sideinfo_data['tags'].apply(str_to_dict)
    sideinfo_data['tag_string'] = sideinfo_data['tags_dict'].apply(lambda x: ' '.join(x.keys()) if isinstance(x, dict) else ' '.join(x))

    tag_list = []
    for tags in sideinfo_data['tag_string']:
        tag_list.extend(tags.split())

    tag_list = list(set(tag_list))
    tag_list = pd.DataFrame(tag_list, columns=['tag'])

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


def song_embedding(sideinfo_data,tag_embedded):
    tag_embedding_dict = tag_embedded.set_index('tag').T.to_dict('dict')
    def str_to_dict(s):
        try:
            return ast.literal_eval(s)
        except ValueError:
            return {}
        
    sideinfo_data['tags_dict'] = sideinfo_data['tags'].apply(str_to_dict)    
    # 노래별 태그 임베딩 평균 계산 함수
    def average_tag_embedding(row):
        # 태그별로 해당하는 감정 점수를 담을 리스트 초기화
        embeddings = []
        for tag in row['tags_dict']:
            if tag in tag_embedding_dict:
                # 해당 태그의 감정 점수를 embeddings 리스트에 추가
                embeddings.append([tag_embedding_dict[tag][emotion] for emotion in emotion_columns])
        
        # embeddings 리스트에 데이터가 있는 경우, 평균값 계산
        if embeddings:
            # 각 감정별로 평균을 계산하여 반환
            return pd.Series(np.mean(embeddings, axis=0), index=emotion_columns)
        else:
            # 해당하는 태그가 없는 경우, 모든 감정 점수를 0으로 설정
            return pd.Series(np.zeros(len(emotion_columns)), index=emotion_columns)

    # 각 노래별로 임베딩 평균 계산 후 새로운 열로 추가
    emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
                    , 'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7']
    for emotion in emotion_columns:
        sideinfo_data[emotion] = sideinfo_data.apply(average_tag_embedding, axis=1)[emotion]

    return sideinfo_data


if __name__ == "__main__":
    sideinfo_data = pd.read_csv('./data/preprocessed_music1.csv', index_col=0)
    tag_list = create_tag_list(sideinfo_data)
    tag_embedded = tag_embedding(tag_list) # 3662개 : 30분
    tag_embedded.to_csv('./data/tag_embedded.csv')
    song_embedded = song_embedding(sideinfo_data,tag_embedded)
    song_embedded.to_csv('./data/song_embedded.csv')



