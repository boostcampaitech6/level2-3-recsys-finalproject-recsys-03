import pandas as pd
# from graphsage_inference import graphsage_inference, GraphSAGE, DotPredictor, Model
from models.content_based_model import combining_track, combining_tag, content_based_model
from models.tag_ranking import tag_ranking_load_data, tag_ranking
import time

def load_data():
    song_embedded = pd.read_csv('../data/song_embedded.csv', index_col=0)
    sideinfo_data = pd.read_csv('../data/preprocessed_music2.csv', index_col=0)
    tag_embedded = pd.read_csv('../data/tag_embedded.csv', index_col=0)
    return song_embedded, sideinfo_data, tag_embedded


def inference(login_user_data, input_tags):
    print(login_user_data)
    print(type(login_user_data))
    start_time = time.time()
    
    '''
    input type example:
    login_user_data = pd.DataFrame({"user_id" : 120328, "track_id" : 49708276}, {"user_id" : 120328, "track_id" : 49708277}, {"user_id" : 120328, "track_id" : 49708358})
    input_tags = "driving, party, upbeat, summer, electronic"
    '''
    
    #recommended_track_id = graphsage_inference(k=1000) # Graphsage 결과값 상위 K개
    #tag_embedding.py로 tag_embedded.csv,song_embedded.csv 생성해주세요
    song_embedded, sideinfo_data, tag_embedded = load_data()

    try: 
        if len(login_user_data['track_id'])>=3: 
            #User의 정보가 있는 경우 : 선호 음악 3개 활용
            user_embedded = combining_track(login_user_data,song_embedded,3)
            recommended_track_id = content_based_model(user_embedded,song_embedded,1000)
            content_time = time.time()
            print(f"content time : {content_time - start_time:.5f} sec")
            recommended_list = tag_ranking_load_data(recommended_track_id,sideinfo_data)  # recommended_track_id에 side info 추가
            recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags, N=10) #Graphsage 결과값에 tag ranking(기준:input tag와 동일한 tag수) 적용
            tag_ranking_time = time.time()
            print(f"tag_ranking time : {tag_ranking_time - content_time:.5f} sec")

    except:
        print("user information not found.")
        #User의 정보가 없는경우 : input tags 사용
        user_embedded = combining_tag(input_tags,tag_embedded)
        recommended_track_id = content_based_model(user_embedded,song_embedded,1000)
        recommended_list = tag_ranking_load_data(recommended_track_id,sideinfo_data) 
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags, N=10)


    # 트랙id, 제목, 아티스트, uri의 리스트 -> 딕셔너리의 리스트 형태로 변환
    recommended_tracks = []
    for i in range(len(recommended_uris)):
        track_info = {
            "id": recommended_ids[i],
            "title": recommended_titles[i],
            "artist": recommended_artists[i],
            "uri": recommended_uris[i]
        }
        recommended_tracks.append(track_info)
    
    return recommended_tracks