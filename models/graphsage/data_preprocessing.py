import copy
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


def load_data(args):
    # user - track interaction
    with open(f'{args.data_dir}{args.interaction_filename}', encoding="utf-8") as f:
        interaction_json = json.load(f)
    user_id = []; track_id = []
    for user_dict in interaction_json:
        user_id.extend([user_dict['user_id']]*len(user_dict['track_id']))
        track_id.extend(user_dict['track_id'])
    interaction = pd.DataFrame({'user_id': user_id, 'track_id': track_id})

    # track 정보
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}', usecols=['track_id','uri','track_name','artist_name','tags'])
    track = [['track_id','uri','track_name','artist_name','tags']]
    
    return interaction, track


def train_valid_test_split(interaction, valid_ratio, test_ratio):
    num_user = len(interaction['user_id'].unique())
    num_valid = int(num_user * valid_ratio)
    num_test = int(num_user * test_ratio)
    num_train = num_user - num_valid - num_test

    user_shuffled = np.random.permutation(interaction['user_id'].unique())
    user_train = user_shuffled[:num_train]
    user_valid = user_shuffled[num_train:num_train+num_valid]
    user_test = user_shuffled[num_train+num_valid:]

    train_interaction = interaction[interaction['user_id'].isin(user_train)]
    valid_interaction = interaction[interaction['user_id'].isin(user_valid)]
    test_interaction = interaction[interaction['user_id'].isin(user_test)]

    return train_interaction, valid_interaction, test_interaction


def mapping_index(args, train_interaction, valid_interaction, test_interaction, track):
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
    for index, track_id in enumerate(sorted(track['track_id'].unique().tolist())):
        mapping_track_index[track_id] = index
        mapping_index_track[index] = track_id
    
    # artist mapping
    mapping_artist_index = dict()
    mapping_index_artist = dict()
    for index, artist_name in enumerate(sorted(track['artist_name'].unique().tolist())):
        mapping_artist_index[artist_name] = index
        mapping_index_artist[index] = artist_name
    
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
    
    return (mapping_user_index, mapping_index_user,
            mapping_track_index, mapping_index_track,
            mapping_artist_index, mapping_index_artist)


def convert_to_graph(args, train_interaction, valid_interaction, test_interaction, track,
                     mapping_user_index, mapping_track_index, mapping_artist_index):
    n_user = len(mapping_user_index)
    n_track = len(mapping_track_index)
    n_artist = len(mapping_artist_index)

    # train data
    train_data = HeteroData()
    train_data['user'].node_id = torch.arange(n_user)
    train_data['track'].node_id = torch.arange(n_track)
    train_data['artist'].node_id = torch.arange(n_artist)
    # node 특성 정보
    # train_data['user'].x = 
    # train_data['track'].x = 
    # train_data['artist'].x = 
    # user-track interaction
    user_id_mapped = [mapping_user_index[i] for i in train_interaction['user_id'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in train_interaction['track_id'].tolist()]
    train_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)
    train_data['user', 'listen', 'track'].edge_index = train_edge_index
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track['artist_name'].tolist()]
    train_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    train_data = ToUndirected()(train_data)

    # valid data
    valid_data = HeteroData()
    valid_data['user'].node_id = torch.arange(n_user)
    valid_data['track'].node_id = torch.arange(n_track)
    valid_data['artist'].node_id = torch.arange(n_artist)
    # node 특성 정보
    # valid_data['user'].x = 
    # valid_data['track'].x = 
    # valid_data['artist'].x = 
    # user-track interaction
    valid_user_id_mapped = [mapping_user_index[i] for i in valid_interaction['user_id'].tolist()]
    valid_track_id_mapped = [mapping_track_index[i] for i in valid_interaction['track_id'].tolist()]
    valid_edge_index = torch.tensor([valid_user_id_mapped, valid_track_id_mapped], dtype=torch.long)
    valid_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track['artist_name'].tolist()]
    valid_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    valid_data = ToUndirected()(valid_data)

    # test data
    test_data = HeteroData()
    test_data['user'].node_id = torch.arange(n_user)
    test_data['track'].node_id = torch.arange(n_track)
    test_data['artist'].node_id = torch.arange(n_artist)
    # node 특성 정보 추가
    # test_data['user'].x = 
    # test_data['track'].x = 
    # test_data['artist'].x = 
    # user-track interaction
    test_user_id_mapped = [mapping_user_index[i] for i in test_interaction['user_id'].tolist()]
    test_track_id_mapped = [mapping_track_index[i] for i in test_interaction['track_id'].tolist()]
    test_edge_index = torch.tensor([test_user_id_mapped, test_track_id_mapped], dtype=torch.long)
    test_data['user', 'listen', 'track'].edge_index = torch.cat([train_edge_index, test_edge_index], dim=1)
    # track-artist matching
    track_id_mapped = [mapping_track_index[i] for i in track['track_id'].tolist()]
    artist_id_mapped = [mapping_artist_index[i] for i in track['artist_name'].tolist()]
    test_data['track', 'sungby', 'artist'].edge_index = torch.tensor([track_id_mapped, artist_id_mapped], dtype=torch.long)
    test_data = ToUndirected()(test_data)

    # serving data 저장
    serving_data = copy.deepcopy(train_data)
    del serving_data['track','rev_listen','user'], serving_data['artist','rev_sungby','track']
    torch.save(serving_data, f'{args.data_dir}serving_data.pt')
    
    return train_data, valid_data, test_data, valid_edge_index, test_edge_index
