import time
import pandas as pd
from models.inference_graphsage import inference_graphsage
from models.content_based_model import combining_track, combining_tag, content_based_model
from models.tag_ranking import tag_ranking_load_data, tag_ranking, genre_export, genre_filtering, filter_by_genre, filter_tags_by_input

def load_data():
    song_embedded = pd.read_csv('../data/song_embedded_hyperpersonalized.csv', index_col=0)
    sideinfo_data = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
    tag_embedded = pd.read_csv('../data/tag_embedded.csv', index_col=0)
    tag_genre_list = pd.read_csv('../data/track_genre.csv')

    return song_embedded, sideinfo_data, tag_embedded, tag_genre_list


def inference(login_user_data, input_tags):
    start_time = time.time()
    
    '''
    #input type example:
    login_user_data = pd.DataFrame({"user_id" : 120328, "track_id" : 49708276}, {"user_id" : 120328, "track_id" : 49708277}, {"user_id" : 120328, "track_id" : 49708358})
    input_tags = "driving, party, upbeat, summer, electronic"
    '''
    
    #tag_embedding.py로 tag_embedded.csv,song_embedded.csv 생성해주세요
    song_embedded, sideinfo_data, tag_embedded, tag_genre_list = load_data()
    
    recommended_tracks = []    # 최종 추천 결과 
    try: 
        # login_user_data에서 interaction이 있는 track은 GraphSAGE로 학습, 없는 track은 Content based model로 학습
        login_user_data = login_user_data.merge(sideinfo_data[['track_id','interaction_exist']], how='left', left_on='track_id', right_on='track_id')
        login_user_data_interaction = login_user_data[login_user_data['interaction_exist'] == 1].drop('interaction_exist', axis=1)
        login_user_data_content = login_user_data[login_user_data['interaction_exist'] == 0].drop('interaction_exist', axis=1)
        


        # GraphSAGE model for interaction data
        interaction_time = time.time()
        print(f"data load time : {interaction_time - start_time:.5f} sec")
        
        recommended_track_id_interaction = inference_graphsage(login_user_data_interaction, k=100)    # Graphsage 결과값 상위 K개
        recommended_list_interaction = tag_ranking_load_data(recommended_track_id_interaction,sideinfo_data)
        input_genres = filter_tags_by_input(sideinfo_data,input_tags)
        recommended_list_filtered = filter_by_genre(recommended_list_interaction, input_genres)


        # Content based model for content data
        content_time = time.time()
        print(f"interaction time : {content_time - interaction_time:.5f} sec")
        
        if len(login_user_data_content['track_id'])>=3: 
            #User의 정보가 있는 경우 : 선호 음악 3개 활용
            user_embedded = combining_track(login_user_data_content,song_embedded,3)
            genre_preferred = genre_export(login_user_data_content,sideinfo_data,tag_genre_list)
            song_filtered = genre_filtering(genre_preferred,song_embedded)
            recommended_track_id_content = content_based_model(user_embedded,song_filtered,100)
        else:
            #User의 정보가 없는경우 : input tags 사용
            user_embedded = combining_tag(input_tags,tag_embedded)
            recommended_track_id_content = content_based_model(user_embedded,song_embedded,100)
        recommended_list_content = tag_ranking_load_data(recommended_track_id_content,sideinfo_data)    # recommended_track_id_content에 side info 추가
        
        tag_ranking_time = time.time()
        print(f"content time : {tag_ranking_time - content_time:.5f} sec")
        
        # tag ranking for interaction model
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list_interaction, input_tags, N=10)
        # 트랙id, 제목, 아티스트, uri의 리스트 -> 딕셔너리의 리스트 형태로 변환
        for i in range(len(recommended_uris)):
            track_info = {
                "id": recommended_ids[i],
                "title": recommended_titles[i],
                "artist": recommended_artists[i],
                "uri": recommended_uris[i]
            }
            recommended_tracks.append(track_info)
        
        # tag ranking for content model
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list_content, input_tags, N=10)
        # 트랙id, 제목, 아티스트, uri의 리스트 -> 딕셔너리의 리스트 형태로 변환
        for i in range(len(recommended_uris)):
            track_info = {
                "id": recommended_ids[i],
                "title": recommended_titles[i],
                "artist": recommended_artists[i],
                "uri": recommended_uris[i]
            }
            recommended_tracks.append(track_info)
    
    except:
        print("user information not found.")
        content_time = time.time()
        
        #User의 정보가 없는경우 : input tags 사용
        user_embedded = combining_tag(input_tags,tag_embedded)
        recommended_track_id_content = content_based_model(user_embedded,song_embedded,100)
        recommended_list = tag_ranking_load_data(recommended_track_id_content,sideinfo_data)

        tag_ranking_time = time.time()
        print(f"content time : {tag_ranking_time - content_time:.5f} sec")
        
        recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags = tag_ranking(recommended_list, input_tags, N=20)
        
        # 트랙id, 제목, 아티스트, uri의 리스트 -> 딕셔너리의 리스트 형태로 변환
        for i in range(len(recommended_uris)):
            track_info = {
                "id": recommended_ids[i],
                "title": recommended_titles[i],
                "artist": recommended_artists[i],
                "uri": recommended_uris[i]
            }
            recommended_tracks.append(track_info)
    
    end_time = time.time()
    print(f"tag_ranking time : {end_time - tag_ranking_time:.5f} sec")
    print(f"total time : {end_time - start_time:.5f} sec")
    
    return recommended_tracks
