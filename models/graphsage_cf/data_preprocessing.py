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
    user['track_id_list'] = user['track_id_list'].apply(eval)
    user_id = []; track_id = []
    for user_id_tmp, track_list_tmp in zip(user['user_id'], user['track_id_list']):
        user_id.extend([user_id_tmp]*len(track_list_tmp))
        track_id.extend(track_list_tmp)
    user_track = pd.DataFrame({'user_id': user_id, 'track_id': track_id})
        
    # user 당 track 20개씩 샘플링
    user_track = user_track.groupby('user_id').apply(lambda x: x.sample(n=20, replace=True) if len(x) > 20 else x).reset_index(drop=True)
    
    # track 데이터
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}')
    
    # track 임베딩
    track['track_emb'] = track['track_emb'].apply(eval)
    track_emb = track[['track_id','track_emb']]
    
    # artist-track interaction 데이터
    artist_track = track[['artist_name','track_id']]
    
    # tag-track interaction 데이터
    track['tag_name_list'] = track['tag_name_list'].apply(eval)
    track_id = []; tag_name = []
    for track_id_tmp, tag_name_list_tmp in zip(track['track_id'], track['tag_name_list']):
        track_id.extend([track_id_tmp]*len(tag_name_list_tmp))
        tag_name.extend(tag_name_list_tmp)
    tag_track = pd.DataFrame({'tag_name': tag_name, 'track_id': track_id})
    
    # artist 임베딩
    artist_emb = pd.read_csv(f'{args.data_dir}{args.artist_filename}')
    artist_emb['artist_emb'] = artist_emb['artist_emb'].apply(eval)
    artist_emb = artist_emb[artist_emb['artist_name'].isin(artist_track['artist_name'].unique())]
    
    # tag 임베딩
    tag_emb = pd.read_csv(f'{args.data_dir}{args.tag_filename}')
    tag_emb['tag_emb'] = tag_emb['tag_emb'].apply(eval)
    tag_emb = tag_emb[tag_emb['tag_name'].isin(tag_track['tag_name'].unique())]
    
    # tag에 artist 포함
    artist_track.columns = ['tag_name','track_id']
    artist_emb.columns = ['tag_name','tag_emb']
    tag_track = pd.concat([tag_track, artist_track])
    tag_emb = pd.concat([tag_emb, artist_emb])
    
    return user_track, tag_track, user_emb, track_emb, tag_emb


def train_valid_test_split(user_track, valid_ratio, test_ratio):
    user_list = user_track['user_id'].unique()
    
    num_user = len(user_list)
    num_valid = int(num_user * valid_ratio)
    num_test = int(num_user * test_ratio)
    num_train = num_user - num_valid - num_test

    user_shuffled = np.random.permutation(user_list)
    user_train = user_shuffled[:num_train]
    user_valid = user_shuffled[num_train:num_train+num_valid]
    user_test = user_shuffled[num_train+num_valid:]

    train_edge = user_track[user_track['user_id'].isin(user_train)]
    valid_edge = user_track[user_track['user_id'].isin(user_valid)]
    test_edge = user_track[user_track['user_id'].isin(user_test)]

    return train_edge, valid_edge, test_edge


def mapping_index(args, train_edge, valid_edge, test_edge, tag_track):
    # user mapping
    mapping_user_index = dict()
    mapping_index_user = dict()
    train_user_list = sorted(train_edge['user_id'].unique().tolist())
    valid_user_list = sorted(valid_edge['user_id'].unique().tolist())
    test_user_list = sorted(test_edge['user_id'].unique().tolist())
    for index, user_id in enumerate(train_user_list + valid_user_list + test_user_list):
        mapping_user_index[user_id] = index
        mapping_index_user[index] = user_id
    
    # track mapping
    mapping_track_index = dict()
    mapping_index_track = dict()
    for index, track_id in enumerate(sorted(tag_track['track_id'].unique().tolist())):
        mapping_track_index[track_id] = index
        mapping_index_track[index] = track_id
    
    # tag mapping
    mapping_tag_index = dict()
    mapping_index_tag = dict()
    for index, tag_name in enumerate(sorted(tag_track['tag_name'].unique().tolist())):
        mapping_tag_index[tag_name] = index
        mapping_index_tag[index] = tag_name
    
    # json 형식으로 저장
    with open(f'{args.data_dir}mapping_{args.model_name}_user_index.json', 'w') as f : 
        json.dump(mapping_user_index, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_index_user.json', 'w') as f : 
        json.dump(mapping_index_user, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_track_index.json', 'w') as f : 
        json.dump(mapping_track_index, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_index_track.json', 'w') as f : 
        json.dump(mapping_index_track, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_tag_index.json', 'w') as f : 
        json.dump(mapping_tag_index, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_index_tag.json', 'w') as f : 
        json.dump(mapping_index_tag, f)
    
    return mapping_user_index, mapping_track_index, mapping_tag_index


def convert_to_graph(args, train_edge, valid_edge, test_edge, tag_track,
                     user_emb, track_emb, tag_emb,
                     mapping_user_index, mapping_track_index, mapping_tag_index):
    n_user = len(mapping_user_index) + 1
    n_track = len(mapping_track_index)
    n_tag = len(mapping_tag_index)

    # node 특성 정보 tensor
    user_emb['user_id'] = user_emb['user_id'].replace(mapping_user_index)    # user_id -> index
    user_emb = user_emb.sort_values('user_id')    # index로 정렬
    x_user = torch.tensor(list(user_emb['user_emb']))
    x_user = torch.vstack([x_user, torch.tensor(np.zeros(args.x_dim), dtype=torch.float32)])    # inference에서 사용할 가상의 user 추가
    track_emb['track_id'] = track_emb['track_id'].replace(mapping_track_index)
    track_emb = track_emb.sort_values('track_id')
    x_track = torch.tensor(list(track_emb['track_emb']))
    tag_emb['tag_name'] = tag_emb['tag_name'].replace(mapping_tag_index)
    tag_emb = tag_emb.sort_values('tag_name')
    x_tag = torch.tensor(list(tag_emb['tag_emb']))

    # tag-track matching 정보
    tagged_tag_id_mapped = [mapping_tag_index[i] for i in tag_track['tag_name'].tolist()]
    tagged_track_id_mapped = [mapping_track_index[i] for i in tag_track['track_id'].tolist()]
    
    # train data
    train_data = HeteroData()
    train_data['user'].node_id = torch.arange(n_user)
    train_data['track'].node_id = torch.arange(n_track)
    train_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    train_data['user'].x = x_user
    train_data['track'].x = x_track
    train_data['tag'].x = x_tag
    # user-track interaction
    user_id_mapped = [mapping_user_index[i] for i in train_edge['user_id'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in train_edge['track_id'].tolist()]
    train_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)
    train_data['user', 'listen', 'track'].edge_index = train_edge_index
    # tag-track matching
    train_data['tag', 'tagged', 'track'].edge_index = torch.tensor([tagged_tag_id_mapped, tagged_track_id_mapped], dtype=torch.long)
    train_data = ToUndirected()(train_data)

    # valid data
    valid_data = HeteroData()
    valid_data['user'].node_id = torch.arange(n_user)
    valid_data['track'].node_id = torch.arange(n_track)
    valid_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    valid_data['user'].x = x_user
    valid_data['track'].x = x_track
    valid_data['tag'].x = x_tag
    # user-track interaction
    valid_user_id_mapped = [mapping_user_index[i] for i in valid_edge['user_id'].tolist()]
    valid_track_id_mapped = [mapping_track_index[i] for i in valid_edge['track_id'].tolist()]
    valid_edge_index = torch.tensor([valid_user_id_mapped, valid_track_id_mapped], dtype=torch.long)
    valid_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
    # tag-track matching
    valid_data['tag', 'tagged', 'track'].edge_index = torch.tensor([tagged_tag_id_mapped, tagged_track_id_mapped], dtype=torch.long)
    valid_data = ToUndirected()(valid_data)

    # test data
    test_data = HeteroData()
    test_data['user'].node_id = torch.arange(n_user)
    test_data['track'].node_id = torch.arange(n_track)
    test_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보 추가
    test_data['user'].x = x_user
    test_data['track'].x = x_track
    test_data['tag'].x = x_tag
    # user-track interaction
    test_user_id_mapped = [mapping_user_index[i] for i in test_edge['user_id'].tolist()]
    test_track_id_mapped = [mapping_track_index[i] for i in test_edge['track_id'].tolist()]
    test_edge_index = torch.tensor([test_user_id_mapped, test_track_id_mapped], dtype=torch.long)
    test_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, test_edge_index], dim=1)
    # tag-track matching
    test_data['tag', 'tagged', 'track'].edge_index = torch.tensor([tagged_tag_id_mapped, tagged_track_id_mapped], dtype=torch.long)
    test_data = ToUndirected()(test_data)

    # serving data 저장
    serving_data = HeteroData()
    serving_data['user'].node_id = torch.arange(n_user)
    serving_data['track'].node_id = torch.arange(n_track)
    serving_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보 추가
    serving_data['user'].x = x_user
    serving_data['track'].x = x_track
    serving_data['tag'].x = x_tag
    # user-track interaction / tag-track matching
    serving_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index, test_edge_index], dim=1)
    serving_data['tag', 'tagged', 'track'].edge_index = torch.tensor([tagged_tag_id_mapped, tagged_track_id_mapped], dtype=torch.long)
    serving_data = ToUndirected()(serving_data)
    # graph data 저장
    torch.save(serving_data, f'{args.data_dir}graph_data_{args.model_name}.pt')
    
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index    # valid_edge_index / test_edge_index : train edge를 포함하지 않는 edge 출력


def data_preprocessing(args):
    # 데이터 로드
    user_track, tag_track, user_emb, track_emb, tag_emb = load_data(args)
    
    # Train-Valid-Test 분할
    train_edge, valid_edge, test_edge = train_valid_test_split(user_track, args.valid_ratio, args.test_ratio)
    
    # 인덱싱
    mapping_user_index, mapping_track_index, mapping_tag_index = mapping_index(args, train_edge, valid_edge, test_edge, tag_track)
    
    # Hetero Graph Data로 변환
    train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index = convert_to_graph(args, train_edge, valid_edge, test_edge, tag_track,
                                                                                                              user_emb, track_emb, tag_emb,
                                                                                                              mapping_user_index, mapping_track_index, mapping_tag_index)
   
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index
