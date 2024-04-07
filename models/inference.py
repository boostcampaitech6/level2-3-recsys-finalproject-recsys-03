import pandas as pd
from models.inference_tag_model import inference_tag_model, load_data_for_tag_model, load_tag_model
from models.inference_cf_model import inference_cf_model, load_data_for_cf_model, update_user_graph_data, load_cf_model
from models.inference_cbf_model import inference_cbf_model, load_data_for_cbf_model, load_cbf_model
from models.filtering import filter_by_genre, filter_tags_by_input, load_info

def load_data():

    sideinfo_data = pd.read_csv('../data/track_inference.csv')

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

    # track list가 없는 유저 (비회원)
    if login_user_data is None:
        # tag_model
        mapping_index_track, mapping_tag_index, graph_data = load_data_for_tag_model()
        embeddings = load_tag_model(graph_data)
        recommended_list = inference_tag_model(100, input_tag_list, mapping_index_track, mapping_tag_index, embeddings)
        
        # genre filtering
        recommended_list_info = load_info(recommended_list, sideinfo_data)  # Track list에 info(title,artist...)추가
        input_genres = filter_tags_by_input(sideinfo_data, input_tags)  # input_genres : input tags에 있는 genre tags
        recommended_list_filtered = filter_by_genre(recommended_list_info, input_genres)  # input_genres에 있는 장르만 남기기
        recommended_playlist = recommended_list_filtered[:20]
        recommended_playlist = recommended_playlist['track_id'].to_list()
        
    # track list가 있는 유저 (회원)
    else:
        # tag_model
        mapping_index_track, mapping_tag_index, graph_data = load_data_for_tag_model()
        embeddings = load_tag_model(graph_data)
        recommended_list = inference_tag_model(100, input_tag_list, mapping_index_track, mapping_tag_index, embeddings)
        
        # genre filtering
        recommended_list_info = load_info(recommended_list, sideinfo_data)
        input_genres = filter_tags_by_input(sideinfo_data, input_tags)
        recommended_list_filtered = filter_by_genre(recommended_list_info, input_genres)
        
        # user가 선호하는 track 분류 (with_interaction / without_interaction)
        login_user_data = login_user_data.merge(sideinfo_data[['track_id','interaction_exist']], how='left', left_on='track_id', right_on='track_id')
        login_user_data_interaction = login_user_data[login_user_data['interaction_exist'] == 1].drop('interaction_exist', axis=1)
        login_user_data_content = login_user_data[login_user_data['interaction_exist'] == 0].drop('interaction_exist', axis=1)
        # 리스트로 변경
        login_user_data_interaction = login_user_data_interaction['track_id'].to_list()
        login_user_data_content = login_user_data_content['track_id'].to_list()

        # tag, genre filtered track 분류 (with_interaction / without_interaction)
        recommended_list_filtered = recommended_list_filtered.merge(sideinfo_data[['track_id','interaction_exist']], how='left', left_on='track_id', right_on='track_id')
        recommended_list_filtered_interaction = recommended_list_filtered[recommended_list_filtered['interaction_exist'] == 1].drop('interaction_exist', axis=1)
        recommended_list_filtered_content = recommended_list_filtered[recommended_list_filtered['interaction_exist'] == 0].drop('interaction_exist', axis=1)
        # 리스트로 변경
        recommended_list_filtered_interaction = recommended_list_filtered_interaction['track_id'].to_list()
        recommended_list_filtered_content = recommended_list_filtered_content['track_id'].to_list()
        
        # track_with_interaction과 track_without_interaction이 균형있게 필터링된 경우
        if len(recommended_list_filtered_interaction) >= 10 and len(recommended_list_filtered_content) >= 10 and len(login_user_data_interaction) >= 1 and len(login_user_data_content) >= 1:
            # cf model
            track_emb, mapping_index_track, mapping_track_index, graph_data = load_data_for_cf_model(login_user_data_interaction)
            updated_graph_data, new_user_id = update_user_graph_data(graph_data, login_user_data_interaction, track_emb, mapping_track_index)
            embeddings = load_cf_model(updated_graph_data)
            recommended_list_cf = inference_cf_model(10, recommended_list_filtered_interaction, mapping_index_track, mapping_track_index, embeddings, new_user_id)
            
            # cbf model
            mapping_index_track, mapping_track_index, graph_data = load_data_for_cbf_model()
            embeddings = load_cbf_model(graph_data)
            recommended_list_cbf = inference_cbf_model(10, login_user_data_content, recommended_list_filtered_content, mapping_index_track, mapping_track_index, embeddings)

            # cf 결과 + cbf 결과
            recommended_list_shuffled = [item for pair in zip(recommended_list_cbf, recommended_list_cf) for item in pair]
            recommended_playlist = recommended_list_shuffled
        
        # track_with_interaction이 주로 필터링된 경우
        elif len(recommended_list_filtered_interaction) >= 20 and len(login_user_data_interaction) >= 1:
            # cf model
            track_emb, mapping_index_track, mapping_track_index, graph_data = load_data_for_cf_model(login_user_data_interaction)
            updated_graph_data, new_user_id = update_user_graph_data(graph_data, login_user_data_interaction, track_emb, mapping_track_index)
            embeddings = load_cf_model(updated_graph_data)
            recommended_list_cf = inference_cf_model(20, recommended_list_filtered_interaction, mapping_index_track, mapping_track_index, embeddings, new_user_id)
            recommended_playlist = recommended_list_cf
        
        # track_without_interaction이 주로 필터링된 경우
        elif len(recommended_list_filtered_content) >= 20 and len(login_user_data_content) >= 1:
            # cbf model
            mapping_index_track, mapping_track_index, graph_data = load_data_for_cbf_model()
            embeddings = load_cbf_model(graph_data)
            recommended_list_cbf = inference_cbf_model(20, login_user_data_content, recommended_list_filtered_content, mapping_index_track, mapping_track_index, embeddings)
            recommended_playlist = recommended_list_cbf
        
        else:
            # user 정보를 사용하지 못하는 경우에 tag_model만 사용
            recommended_playlist = recommended_list_filtered[:20]
            recommended_playlist = recommended_playlist['track_id'].to_list()

    # recommended track list -> dictionary
    recommended_playlist_info = load_info(recommended_playlist, sideinfo_data)
    for i in range(len(recommended_playlist_info)):
        track_info = {
            "id": int(recommended_playlist_info.loc[i]['track_id']),
            "title": recommended_playlist_info.loc[i]['track_name'],
            "artist": recommended_playlist_info.loc[i]['artist_name'],
            "uri": recommended_playlist_info.loc[i]['uri']
        }
        recommended_playlist_dict.append(track_info)

    return recommended_playlist_dict
