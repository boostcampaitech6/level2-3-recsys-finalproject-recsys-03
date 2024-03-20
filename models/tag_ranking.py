import pandas as pd
import ast
import re

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []
    
def tag_ranking_load_data(recommended_track_id,sideinfo_data):
    sideinfo_data['tag_string'] = sideinfo_data['tags'].apply(lambda x: ' '.join(string_to_list(x)))   
    # recommended_list : recommended_track_id에 side info 추가
    recommended_track = pd.DataFrame(recommended_track_id, columns=['track_id'])
    recommended_list = pd.merge(recommended_track, sideinfo_data[['track_id', 'track_name', 'artist_name', 'tag_string', 'uri']], on='track_id', how='left')
    return recommended_list
   
    
def tag_ranking(recommended_list, input_tags, N):
    # 입력 태그를 쉼표를 기준으로 분할하여 단어 추출
    input_words = [word.strip() for word in input_tags.split(',')]
    # 각 문서에 대해 입력 태그와 일치하는 태그만 선택하여 추출
    selected_tags = []
    for track_tags in recommended_list['tag_string']:
        selected_tags.append(' '.join(word for word in input_words if word in track_tags.split()))
    # 각 문서에 대해 입력 태그와 일치하는 단어의 수 계산
    similarity_scores = [(track_id, track_name, artist_name, uri, selected_tag, sum(word in selected_tag.split() for word in input_words))
                         for track_id, track_name, artist_name, uri, selected_tag in zip(recommended_list['track_id'], recommended_list['track_name'], recommended_list['artist_name'], recommended_list['uri'], selected_tags)]
    # 유사도를 기준으로 내림차순 정렬
    similarity_scores.sort(key=lambda x: x[5], reverse=True)
    # 상위 N개의 추천 곡 정보를 리스트로 저장
    recommended_tracks = similarity_scores[:N]
    # 추천 곡 정보를 각각의 리스트로 분리하여 반환
    recommended_ids = [track[0] for track in recommended_tracks]
    recommended_titles = [track[1] for track in recommended_tracks]
    recommended_artists = [track[2] for track in recommended_tracks]
    recommended_uris = [track[3] for track in recommended_tracks]
    recommended_selected_tags = [track[4] for track in recommended_tracks]

    return recommended_ids, recommended_titles, recommended_artists, recommended_uris, recommended_selected_tags

def genre_export(login_user_data_content,sideinfo_data,tag_genre_list):
    login_user_data_content_sideinfo = tag_ranking_load_data(login_user_data_content,sideinfo_data)
    login_user_data_content_sideinfo['tag_list'] = login_user_data_content_sideinfo['tag_string'].apply(lambda x: x.split())

    genre_tags_set = set(tag_genre_list['Genre Tags'].str.lower().apply(lambda x: x.split()).explode())

    # Filter tag_list in login_user_data_content_sideinfo based on genre_tags_set
    login_user_data_content_sideinfo['genre_list'] = login_user_data_content_sideinfo['tag_list'].apply(
        lambda tags: [tag for tag in tags if tag in genre_tags_set]
    )

    genre_preferred = list(set(genre for sublist in login_user_data_content_sideinfo['genre_list'] for genre in sublist))
    return genre_preferred

def genre_filtering(genre_preferred,song_embedded):
    pattern = '|'.join([fr"\'{genre}\'(?=[^a-zA-Z]|$)" for genre in genre_preferred])

    filtered_songs = song_embedded[song_embedded['tags'].str.contains(pattern, flags=re.IGNORECASE, na=False)]
    return filtered_songs
