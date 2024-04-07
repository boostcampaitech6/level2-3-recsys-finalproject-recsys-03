import pandas as pd
import numpy as np
import ast

def make_track(song_embedded):
    columns_embedded = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                    'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7' , 
                    'danceability', 'energy', 'key', 'loudness','speechiness', 'acousticness', 'instrumentalness',  'valence', 'tempo',  
                    ]   
    song_embedded['track_emb']=song_embedded[columns_embedded].apply(list, axis=1)
    columns = ['track_id', 'track_name', 'artist_name', 'tags', 'track_emb']
    track = song_embedded[columns]
    track.reset_index(drop=True, inplace=True)
    new_column_names = ['track_id', 'track_name', 'artist_name', 'tag_name_list', 'track_emb']
    track.columns = new_column_names
    return track

# Hyper-personalized Model - Track.csv
song_embedded = pd.read_csv('../data/song_embedded_hyperpersonalized.csv', index_col=0)
track = make_track(song_embedded)
track.to_csv('../data/track.csv',index=False)

# Personalized Model : track_with_interaction, track_without_interaction.csv
sideinfo = pd.read_csv('../data/preprocessed_music5.csv', index_col=0)
song_embedded_personalized = pd.read_csv('../data/song_embedded_personalized.csv', index_col=0)
track_genre = make_track(song_embedded_personalized)
track_genre = track_genre.merge(sideinfo[['track_id','interaction_exist']],how='left', on='track_id') # tag_name_list에 Genre만 유지
track_genre.to_csv('../data/track_genre.csv',index=False)

track_with_interaction = track_genre[track_genre['interaction_exist']==1]
track_without_interaction = track_genre[track_genre['interaction_exist']==0]
track_with_interaction.to_csv('../data/track_with_interaction.csv',index=False)
track_without_interaction.to_csv('../data/track_without_interaction.csv',index=False)


def make_artist_context(track):
    artist_context = pd.DataFrame(list(set(track['artist_name'])), columns=['artist_name'])
    artist_context['artist_emb'] = None 
    artist_context = artist_context.set_index('artist_name')

    for artist in artist_context.index:
        tracks = track[track['artist_name'] == artist]
        if not tracks.empty:
            avg_emb = np.mean([np.array(ast.literal_eval(x)) for x in tracks['track_emb']], axis=0)
            artist_context.at[artist, 'artist_emb'] = avg_emb.tolist()
    artist_context['artist_emb'] = artist_context['artist_emb'].apply(lambda emb_list: [round(x, 4) for x in emb_list])


    artist_context.reset_index(inplace=True)
    return artist_context

# Hyper-personalized Model - Artist_context
track = pd.read_csv('../data/track.csv', index_col=0)
artist_context = make_artist_context(track)
artist_context.to_csv('../data/artist_context.csv',index=False)

# Personalized Model - Artist_genre
track_genre = pd.read_csv('../data/track_genre.csv', index_col=0)
artist_genre = make_artist_context(track_genre)
artist_genre.to_csv('../data/artist_genre.csv',index=False)


def tag_context(track,song_embedded):
    columns_embedded = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7' , 
                'danceability', 'energy', 'key', 'loudness','speechiness', 'acousticness', 'instrumentalness',  'valence', 'tempo',  
                    ]  
    track.reset_index(inplace=True)
    track['tag_name_list'] = track['tag_name_list'].apply(eval)

    song_emb = pd.DataFrame()
    song_emb['track_id'] = song_embedded['track_id']
    song_emb['tag_emb'] = song_embedded[columns_embedded].apply(lambda row: row.tolist(), axis=1)

    track_id = []; tag_name = []
    for id_tmp, list_tmp in zip(track['track_id'], track['tag_name_list']):
        track_id.extend([id_tmp]*len(list_tmp))
        tag_name.extend(list_tmp)

    track_df = pd.DataFrame({'track_id': track_id, 'tag_name': tag_name})
    track_df = track_df.merge(song_emb, how='left', on='track_id')
    #track_df.to_csv('../data/track_df.csv') : 여기서는 문제없이 모든 태그에 emb값이 붙음
    def list_avg(group):
        avg_list = [round(sum(x) / len(x), 4) for x in zip(*group)]
        return avg_list

    tag_context = track_df.groupby('tag_name')['tag_emb'].apply(list_avg).reset_index(name='tag_emb')
    return tag_context

# List 확인
def create_tag_list(track):
    track['tag_name_list'] = track['tag_name_list'].apply(eval)
    tags_set = set()
    for tags in track['tag_name_list']:
        tags_set.update(tags)

    tag_list = pd.DataFrame(tags_set, columns=['tag'])
    return tag_list

track_tag_list = create_tag_list(track)


# Hyper-personalized Model - Tag_context
track = pd.read_csv('../data/track.csv', index_col=0)
song_embedded = pd.read_csv('../data/song_embedded_hyperpersonalized.csv', index_col=0)
tag_context = tag_context(track,song_embedded)
tag_context.to_csv('../data/tag_context.csv',index=False)

# Personalized Model - Tag_genre
track_genre = pd.read_csv('../data/track_genre.csv', index_col=0)
song_embedded_genre = pd.read_csv('../data/song_embedded_personalized.csv', index_col=0)
tag_genre = tag_context(track_genre,song_embedded_genre)
tag_genre.to_csv('../data/tag_genre.csv',index=False)


def user(user_data,song_embedded):
    columns_embedded = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 
                    'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7' , 
                    'danceability', 'energy', 'key', 'loudness','speechiness', 'acousticness', 'instrumentalness',  'valence', 'tempo',  
                    ]   
    track_id_list = user_data.groupby('user_id')['track_id'].apply(list).reset_index(name='track_id_list')

    song_emb = pd.DataFrame()
    song_emb['track_id'] = song_embedded['track_id']
    song_emb['tag_emb'] = song_embedded[columns_embedded].apply(lambda row: row.tolist(), axis=1)

    user_data = user_data.merge(song_emb, how='left', on='track_id')
    def list_avg(group):
        avg_list = [round(sum(x) / len(x), 4) for x in zip(*group)]
        return avg_list

    user = user_data.groupby('user_id')['tag_emb'].apply(list_avg).reset_index(name='user_emb')
    user = track_id_list.merge(user, how='left', on='user_id')
    return user

# Personalized Model - user.csv
user_data = pd.read_csv('../data/interaction.csv',  index_col=0)
song_embedded_genre = pd.read_csv('../data/song_embedded_personalized.csv', index_col=0)
user = user(user_data,song_embedded_genre)
user.to_csv('../data/user.csv',index=False)

