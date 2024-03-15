import pandas as pd
from graphsage_inference import graphsage_inference, GraphSAGE, DotPredictor, Model
from content_based_model import combining_track, combining_tag, content_based_model
from tag_ranking import tag_ranking_load_data, tag_ranking

def load_data():
    login_user_data = {
    "user_id": 120322,
    "track_id": [
        49708132,
        49708373,
        49708211,
        49711010,
        49707378,
        49709489
        ]
    }
    song_embedded = pd.read_csv('../data/song_embedded.csv', index_col=0)
    sideinfo_data = pd.read_csv('../data/preprocessed_music1.csv', index_col=0)
    input_tags = "driving, party, upbeat, summer, electronic"
    tag_embedded = pd.read_csv('../data/tag_embedded.csv', index_col=0)
    return login_user_data, song_embedded, sideinfo_data, input_tags, tag_embedded

def ouput_recommended_tracks(recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags):
    print("Recommended Track IDs:", recommended_ids)
    print("Recommended Titles:", recommended_titles)
    print("Recommended Artists:", recommended_artists)
    print("Recommended URIs:", recommended_uris)
    print("Recommended Selected Tags:", recommended_selected_tags) #input tag와 동일한 tags

if __name__ == "__main__":
    #recommended_track_id = graphsage_inference(k=1000) # Graphsage 결과값 상위 K개
    #tag_embedding.py로 tag_embedded.csv,song_embedded.csv 생성해주세요
    login_user_data, song_embedded, sideinfo_data, input_tags, tag_embedded = load_data()

    if len(login_user_data['track_id'])>=3: #User의 정보가 있는 경우 : 선호 음악 3개 활용
        user_embedded = combining_track(login_user_data,song_embedded,3)
        recommended_track_id = content_based_model(user_embedded,song_embedded,1000)
        recommended_list = tag_ranking_load_data(recommended_track_id,sideinfo_data)  # recommended_track_id에 side info 추가
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags,N=10) #Graphsage 결과값에 tag ranking(기준:input tag와 동일한 tag수) 적용
    else: #User의 정보가 없는경우 : input tags 사용
        user_embedded = combining_tag(input_tags,tag_embedded)
        recommended_track_id = content_based_model(user_embedded,song_embedded,1000)
        recommended_list = tag_ranking_load_data(recommended_track_id,sideinfo_data) 
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags,N=10)

    ouput_recommended_tracks(recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags)
