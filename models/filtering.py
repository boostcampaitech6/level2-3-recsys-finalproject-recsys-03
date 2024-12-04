import pandas as pd
import ast
import re

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []

def load_info(recommended_track_id,sideinfo_data):
    #sideinfo_data['tag_string'] = sideinfo_data['tags'].apply(lambda x: ' '.join(string_to_list(x)))   
    # recommended_list : recommended_track_id에 side info 추가
    recommended_track = pd.DataFrame(recommended_track_id, columns=['track_id'])
    recommended_list = pd.merge(recommended_track, sideinfo_data[['track_id', 'track_name', 'artist_name', 'tag_name_list', 'genres', 'uri']], on='track_id', how='left')
    return recommended_list
   
    
def tag_ranking(recommended_list, input_tags, N):
    input_words = [word.strip() for word in input_tags.split(',')]
    selected_tags = []
    for track_tags in recommended_list['tag_string']:
        selected_tags.append(' '.join(word for word in input_words if word in track_tags.split()))
    similarity_scores = [(track_id, track_name, artist_name, uri, selected_tag, sum(word in selected_tag.split() for word in input_words))
                         for track_id, track_name, artist_name, uri, selected_tag in zip(recommended_list['track_id'], recommended_list['track_name'], recommended_list['artist_name'], recommended_list['uri'], selected_tags)]
    similarity_scores.sort(key=lambda x: x[5], reverse=True)
    recommended_tracks = similarity_scores[:N]

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

    login_user_data_content_sideinfo['genre_list'] = login_user_data_content_sideinfo['tag_list'].apply(
        lambda tags: [tag for tag in tags if tag in genre_tags_set]
    )

    genre_preferred = list(set(genre for sublist in login_user_data_content_sideinfo['genre_list'] for genre in sublist))
    return genre_preferred

def genre_filtering(genre_preferred,song_embedded):
    pattern = '|'.join([fr"\'{genre}\'(?=[^a-zA-Z]|$)" for genre in genre_preferred])

    filtered_songs = song_embedded[song_embedded['tags'].str.contains(pattern, flags=re.IGNORECASE, na=False)]
    return filtered_songs

# input tags
def filter_by_genre(song_list, input_genres):
    if not input_genres:
        return song_list
    song_list['genres_list'] = song_list['genres'].apply(eval)
    filtered_list = song_list[song_list['genres_list'].apply(lambda genres: any(genre in genres for genre in input_genres))]
    return filtered_list

def create_genre_list(sideinfo_data,col):
    sideinfo_data[col] = sideinfo_data[col].apply(eval)
    tags_set = set()
    for tags in sideinfo_data[col]:
        tags_set.update(tags)

    tag_list = pd.DataFrame(tags_set, columns=[col])
    return tag_list

def filter_tags_by_input(sideinfo_data, input_tags):
    input_tags_list = input_tags.split(', ')
    unique_genres_df = create_genre_list(sideinfo_data, 'genres')
    filtered_tags = unique_genres_df[unique_genres_df['genres'].isin(input_tags_list)]
    filtered_tags_list = filtered_tags['genres'].tolist()
    return filtered_tags_list