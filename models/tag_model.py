from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast

# 데이터 불러오기
data_path = 'data/'
interaction = pd.read_csv(data_path + "interaction.csv", index_col=0)
sideinfo_tag = pd.read_csv(data_path + "sideinfo.csv", index_col=0)

# Tag_string 생성
def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return {}

sideinfo_tag['tag_string'] = sideinfo_tag['tags'].apply(lambda x: ' '.join(str_to_dict(x).keys()))

# TF-IDF 벡터라이저 초기화 및 태그 데이터 변환
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sideinfo_tag['tag_string'])

def recommend_songs(input_tags, N=5):
    # 입력 태그 TF-IDF 벡터화
    input_vec = tfidf_vectorizer.transform([input_tags])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix)

    # 가장 유사한 노래 찾기 (상위 N개)
    similar_indices = cosine_sim.argsort()[0][-N-1:-1][::-1]

    # 추천된 노래의 track_name 출력
    recommended_track_ids = list(sideinfo_tag.iloc[similar_indices]['track_name'].values)

    return recommended_track_ids

if __name__ == "__main__":
    # 사용자로부터 input taga 받기
    input_tags = "pop Love"
    # 입력된 태그로 노래 추천
    print(f"추천된 노래의 Track ID: {recommend_songs(input_tags)}")
