import pandas as pd
from models.inference_tag_model import inference_tag_model, load_data_for_tag_model, load_tag_model
from models.inference_cf_model import inference_cf_model, load_data_for_cf_model, update_user_graph_data, load_cf_model
from models.inference_cbf_model import inference_cbf_model, load_data_for_cbf_model, load_cbf_model
from models.filtering import filter_by_genre, filter_tags_by_input, load_info

def load_data():

    sideinfo_data = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)

    return sideinfo_data

def inference(login_user_data, input_tags):   
    
    '''
    #input type example:
    login_user_data = pd.DataFrame({"user_id" : 120328, "track_id" : 49708276}, {"user_id" : 120328, "track_id" : 49708277}, {"user_id" : 120328, "track_id" : 49708358})
    input_tags = "pop, party, upbeat, summer, electronic"
    '''
    sideinfo_data = load_data()
    input_tag_list = input_tags.split(', ')
    recommended_playlist_dict = []  # 최종 추천 결과

    if login_user_data.empty or login_user_data is None:  # track list가 없는 유저 (비회원)
        recommended_list = inference_tag_model(k=100, input_tag_list=input_tag_list)
        recommended_list_info = load_info(recommended_list, sideinfo_data)  # Track list에 info(title,artist...)추가
        input_genres = filter_tags_by_input(sideinfo_data, input_tags)  # input_genres : input tags에 있는 genre tags
        recommended_list_filtered = filter_by_genre(recommended_list_info, input_genres)  # input_genres에 있는 장르만 남기기
        recommended_playlist = recommended_list_filtered[:20]

    else:  # track list가 있는 유저 (회원)
        login_user_data = login_user_data.merge(sideinfo_data[['track_id','interaction_exist']], how='left', left_on='track_id', right_on='track_id')
        login_user_data_interaction = login_user_data[login_user_data['interaction_exist'] == 1].drop('interaction_exist', axis=1)
        login_user_data_content = login_user_data[login_user_data['interaction_exist'] == 0].drop('interaction_exist', axis=1)

        recommended_list = inference_tag_model(k=100, input_tag_list=input_tag_list)
        recommended_list_info = load_info(recommended_list, sideinfo_data)
        input_genres = filter_tags_by_input(sideinfo_data, input_tags)
        recommended_list_filtered = filter_by_genre(recommended_list_info, input_genres)
        
        if not login_user_data_interaction.empty and not login_user_data_content.empty:  # track_with_interaction과 track_without_interaction이 동시에 있는 유저
            recommended_list_cf = inference_cf_model(k=10, candidate_track_list=recommended_list_filtered)
            recommended_list_cbf = inference_cbf_model(k=10, candidate_track_list=recommended_list_filtered)
            recommended_list_shuffled = [item for pair in zip(recommended_list_cbf, recommended_list_cf) for item in pair]
            recommended_playlist = recommended_list_shuffled

        elif login_user_data_interaction.empty:  # track_without_interaction만 있는 유저
            recommended_list_cbf = inference_cbf_model(k=20, candidate_track_list=recommended_list_filtered)
            recommended_playlist = recommended_list_cbf

        elif login_user_data_content.empty:  # track_with_interaction만 있는 유저
            recommended_list_cf = inference_cf_model(k=20, candidate_track_list=recommended_list_filtered)
            recommended_playlist = recommended_list_cf


    recommended_playlist_info = load_info(recommended_playlist, sideinfo_data)
    for i in range(len(recommended_playlist_info)):
        track_info = {
            "id": recommended_playlist_info.loc[i]['track_id'],
            "title": recommended_playlist_info.loc[i]['track_name'],
            "artist": recommended_playlist_info.loc[i]['artist_name'],
            "uri": recommended_playlist_info.loc[i]['uri']
        }
        recommended_playlist_dict.append(track_info)

    return recommended_playlist_dict

