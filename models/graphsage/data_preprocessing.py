import copy
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


def load_data(args):
    # user 데이터
    user = pd.read_csv(f'{args.data_dir}{args.user_filename}')
    
    # user 임베딩
    user['user_emb'] = user['user_emb'].apply(eval)
    user_emb = user[['user_id','user_emb']]
    
    # user-track interaction 데이터
    user['track_ids'] = user['track_ids'].apply(eval)
    user_id = []; track_id = []
    for user_id_tmp, track_list_tmp in zip(user['user_id'], user['track_ids']):
        user_id.extend([user_id_tmp]*len(track_list_tmp))
        track_id.extend(track_list_tmp)
    user_track = pd.DataFrame({'user_id': user_id, 'track_id': track_id})
    
    # track 데이터
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}')
    
    # track 임베딩
    track['track_emb'] = track['track_emb'].apply(eval)
    track_emb = track[['track_id','track_emb']]
    
    # track-artist interaction 데이터
    track_artist = track[['track_id','artist_name']]
    
    # track-tag interaction 데이터
    track['tags'] = track['tags'].apply(eval)
    track_id = []; tag_name = []
    for track_id_tmp, tags_list_tmp in zip(track['track_id'], track['tags']):
        track_id.extend([track_id_tmp]*len(tags_list_tmp))
        tag_name.extend(tags_list_tmp)
    track_tag = pd.DataFrame({'track_id': track_id, 'tag_name': tag_name})
    
    # artist 임베딩
    artist_emb = pd.read_csv(f'{args.data_dir}{args.artist_filename}')
    artist_emb['artist_emb'] = artist_emb['artist_emb'].apply(eval)
    
    # tag 임베딩
    tag_emb = pd.read_csv(f'{args.data_dir}{args.tag_filename}')
    tag_emb['tag_emb'] = tag_emb['tag_emb'].apply(eval)
    
    return user_track, track_artist, track_tag, user_emb, track_emb, artist_emb, tag_emb


def train_valid_test_split(user_track, valid_ratio, test_ratio):
    num_user = len(user_track['user_id'].unique())
    num_valid = int(num_user * valid_ratio)
    num_test = int(num_user * test_ratio)
    num_train = num_user - num_valid - num_test

    user_shuffled = np.random.permutation(user_track['user_id'].unique())
    user_train = user_shuffled[:num_train]
    user_valid = user_shuffled[num_train:num_train+num_valid]
    user_test = user_shuffled[num_train+num_valid:]

    train_interaction = user_track[user_track['user_id'].isin(user_train)]
    valid_interaction = user_track[user_track['user_id'].isin(user_valid)]
    test_interaction = user_track[user_track['user_id'].isin(user_test)]

    return train_interaction, valid_interaction, test_interaction


def mapping_index(args, train_interaction, valid_interaction, test_interaction, track_artist, track_tag):
    # user mapping
    mapping_user_index = dict()
    mapping_index_user = dict()
    train_user_list = sorted(train_interaction['user_id'].unique().tolist())
    valid_user_list = sorted(valid_interaction['user_id'].unique().tolist())
    test_user_list = sorted(test_interaction['user_id'].unique().tolist())
    for index, user_id in enumerate(train_user_list + valid_user_list + test_user_list):
        mapping_user_index[user_id] = index
        mapping_index_user[index] = user_id
    
    # track mapping
    mapping_track_index = dict()
    mapping_index_track = dict()
    for index, track_id in enumerate(sorted(track_artist['track_id'].unique().tolist())):
        mapping_track_index[track_id] = index
        mapping_index_track[index] = track_id
    
    # artist mapping
    mapping_artist_index = dict()
    mapping_index_artist = dict()
    for index, artist_name in enumerate(sorted(track_artist['artist_name'].unique().tolist())):
        mapping_artist_index[artist_name] = index
        mapping_index_artist[index] = artist_name
    
    # tag mapping
    mapping_tag_index = dict()
    mapping_index_tag = dict()
    for index, tag_name in enumerate(sorted(track_tag['tag_name'].unique().tolist())):
        mapping_tag_index[tag_name] = index
        mapping_index_tag[index] = tag_name
    
    # json 형식으로 저장
    with open(f'{args.data_dir}mapping_user_index.json', 'w') as f : 
        json.dump(mapping_user_index, f)
    with open(f'{args.data_dir}mapping_index_user.json', 'w') as f : 
        json.dump(mapping_index_user, f)
    with open(f'{args.data_dir}mapping_track_index.json', 'w') as f : 
        json.dump(mapping_track_index, f)
    with open(f'{args.data_dir}mapping_index_track.json', 'w') as f : 
        json.dump(mapping_index_track, f)
    with open(f'{args.data_dir}mapping_artist_index.json', 'w') as f : 
        json.dump(mapping_artist_index, f)
    with open(f'{args.data_dir}mapping_index_artist.json', 'w') as f : 
        json.dump(mapping_index_artist, f)
    with open(f'{args.data_dir}mapping_tag_index.json', 'w') as f : 
        json.dump(mapping_tag_index, f)
    with open(f'{args.data_dir}mapping_index_tag.json', 'w') as f : 
        json.dump(mapping_index_tag, f)
    
    return mapping_user_index, mapping_track_index, mapping_artist_index, mapping_tag_index


def convert_to_graph(args, train_interaction, valid_interaction, test_interaction, track_artist, track_tag,
                     user_emb, track_emb, artist_emb, tag_emb,
                     mapping_user_index, mapping_track_index, mapping_artist_index, mapping_tag_index):
    n_user = len(mapping_user_index)
    n_track = len(mapping_track_index)
    n_artist = len(mapping_artist_index)
    n_tag = len(mapping_tag_index)

    # node 특성 정보 tensor
    user_emb['user_id'] = user_emb['user_id'].replace(mapping_user_index)    # user_id -> index
    user_emb = user_emb.sort_values('user_id')    # index로 정렬
    x_user = torch.tensor(list(user_emb['user_emb']))
    track_emb['track_id'] = track_emb['track_id'].replace(mapping_track_index)
    track_emb = track_emb.sort_values('track_id')
    x_track = torch.tensor(list(track_emb['track_emb']))
    artist_emb['artist_name'] = artist_emb['artist_name'].replace(mapping_artist_index)
    artist_emb = artist_emb.sort_values('artist_name')
    x_artist = torch.tensor(list(artist_emb['artist_emb']))
    tag_emb['tag_name'] = tag_emb['tag_name'].replace(mapping_tag_index)
    tag_emb = tag_emb.sort_values('tag_name')
    x_tag = torch.tensor(list(tag_emb['tag_emb']))

    # train data
    train_data = HeteroData()
    train_data['user'].node_id = torch.arange(n_user)
    train_data['track'].node_id = torch.arange(n_track)
    train_data['artist'].node_id = torch.arange(n_artist)
    train_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    train_data['user'].x = x_user
    train_data['track'].x = x_track
    train_data['artist'].x = x_artist
    train_data['tag'].x = x_tag
    # user-track interaction
    user_id_mapped = [mapping_user_index[i] for i in train_interaction['user_id'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in train_interaction['track_id'].tolist()]
    train_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)
    train_data['user', 'listen', 'track'].edge_index = train_edge_index
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track_artist['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track_artist['artist_name'].tolist()]
    train_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    # track-tag matching
    track_id_mapped = [mapping_track_index[i] for i in track_tag['track_id'].tolist()]
    tag_id_mapped = [mapping_tag_index[i] for i in track_tag['tag_name'].tolist()]
    train_data['track', 'tagged', 'tag'].edge_index = torch.tensor([track_id_mapped, tag_id_mapped], dtype=torch.long)
    train_data = ToUndirected()(train_data)

    # valid data
    valid_data = HeteroData()
    valid_data['user'].node_id = torch.arange(n_user)
    valid_data['track'].node_id = torch.arange(n_track)
    valid_data['artist'].node_id = torch.arange(n_artist)
    valid_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    valid_data['user'].x = x_user
    valid_data['track'].x = x_track
    valid_data['artist'].x = x_artist
    valid_data['tag'].x = x_tag
    # user-track interaction
    valid_user_id_mapped = [mapping_user_index[i] for i in valid_interaction['user_id'].tolist()]
    valid_track_id_mapped = [mapping_track_index[i] for i in valid_interaction['track_id'].tolist()]
    valid_edge_index = torch.tensor([valid_user_id_mapped, valid_track_id_mapped], dtype=torch.long)
    valid_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track_artist['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track_artist['artist_name'].tolist()]
    valid_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    # track-tag matching
    track_id_mapped = [mapping_track_index[i] for i in track_tag['track_id'].tolist()]
    tag_id_mapped = [mapping_tag_index[i] for i in track_tag['tag_name'].tolist()]
    valid_data['track', 'tagged', 'tag'].edge_index = torch.tensor([track_id_mapped, tag_id_mapped], dtype=torch.long)
    valid_data = ToUndirected()(valid_data)

    # test data
    test_data = HeteroData()
    test_data['user'].node_id = torch.arange(n_user)
    test_data['track'].node_id = torch.arange(n_track)
    test_data['artist'].node_id = torch.arange(n_artist)
    test_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보 추가
    test_data['user'].x = x_user
    test_data['track'].x = x_track
    test_data['artist'].x = x_artist
    test_data['tag'].x = x_tag
    # user-track interaction
    test_user_id_mapped = [mapping_user_index[i] for i in test_interaction['user_id'].tolist()]
    test_track_id_mapped = [mapping_track_index[i] for i in test_interaction['track_id'].tolist()]
    test_edge_index = torch.tensor([test_user_id_mapped, test_track_id_mapped], dtype=torch.long)
    test_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, test_edge_index], dim=1)
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track_artist['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track_artist['artist_name'].tolist()]
    test_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    # track-tag matching
    track_id_mapped = [mapping_track_index[i] for i in track_tag['track_id'].tolist()]
    tag_id_mapped = [mapping_tag_index[i] for i in track_tag['tag_name'].tolist()]
    test_data['track', 'tagged', 'tag'].edge_index = torch.tensor([track_id_mapped, tag_id_mapped], dtype=torch.long)
    test_data = ToUndirected()(test_data)

    # serving data 저장
    serving_data = copy.deepcopy(train_data)    # serving data에 train 정보 포함
    del serving_data['track','rev_listen','user'], serving_data['artist','rev_sungby','track'], serving_data['tag','rev_tagged','track']    # inference에서 데이터 추가 후에 양방향으로 변환
    torch.save(serving_data, f'{args.data_dir}{args.graph_filename}')
    
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index    # valid_edge_index / test_edge_index : train edge를 포함하지 않는 edge 출력


def data_preprocessing(args):
    # 데이터 로드
    user_track, track_artist, track_tag, user_emb, track_emb, artist_emb, tag_emb = load_data(args)
    
    # Train-Valid-Test 분할
    train_interaction, valid_interaction, test_interaction = train_valid_test_split(user_track, args.valid_ratio, args.test_ratio)
    
    # 인덱싱
    mapping_user_index, mapping_track_index, mapping_artist_index, mapping_tag_index = mapping_index(args, train_interaction, valid_interaction, test_interaction, track_artist, track_tag)
    
    # Hetero Graph Data로 변환
    train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index = convert_to_graph(args, train_interaction, valid_interaction, test_interaction, track_artist, track_tag,
                                                                                                              user_emb, track_emb, artist_emb, tag_emb,
                                                                                                              mapping_user_index, mapping_track_index, mapping_artist_index, mapping_tag_index)
    
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index