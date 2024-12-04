import json
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


def load_data(args):
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
    
    return tag_track, track_emb, tag_emb


def train_valid_test_split(tag_track, valid_ratio, test_ratio):
    num_edge = len(tag_track)
    num_valid = int(num_edge * valid_ratio)
    num_test = int(num_edge * test_ratio)
    num_train = num_edge - num_valid - num_test

    edge_shuffled = tag_track.sample(frac=1)
    train_edge = edge_shuffled[:num_train]
    valid_edge = edge_shuffled[num_train:num_train+num_valid]
    test_edge = edge_shuffled[num_train+num_valid:]

    return train_edge, valid_edge, test_edge


def mapping_index(args, tag_track):
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
    with open(f'{args.data_dir}mapping_{args.model_name}_track_index.json', 'w') as f : 
        json.dump(mapping_track_index, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_index_track.json', 'w') as f : 
        json.dump(mapping_index_track, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_tag_index.json', 'w') as f : 
        json.dump(mapping_tag_index, f)
    with open(f'{args.data_dir}mapping_{args.model_name}_index_tag.json', 'w') as f : 
        json.dump(mapping_index_tag, f)
    
    return mapping_track_index, mapping_tag_index


def convert_to_graph(args, train_edge, valid_edge, test_edge,
                     track_emb, tag_emb,
                     mapping_track_index, mapping_tag_index):
    n_track = len(mapping_track_index)
    n_tag = len(mapping_tag_index)

    # node 특성 정보 tensor
    track_emb['track_id'] = track_emb['track_id'].replace(mapping_track_index)
    track_emb = track_emb.sort_values('track_id')
    x_track = torch.tensor(list(track_emb['track_emb']))
    tag_emb['tag_name'] = tag_emb['tag_name'].replace(mapping_tag_index)
    tag_emb = tag_emb.sort_values('tag_name')
    x_tag = torch.tensor(list(tag_emb['tag_emb']))

    # train data
    train_data = HeteroData()
    train_data['track'].node_id = torch.arange(n_track)
    train_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    train_data['track'].x = x_track
    train_data['tag'].x = x_tag
    # tag-track matching
    tag_id_mapped = [mapping_tag_index[i] for i in train_edge['tag_name'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in train_edge['track_id'].tolist()]
    train_edge_index = torch.tensor([tag_id_mapped, track_id_mapped], dtype=torch.long)
    train_data['tag', 'tagged', 'track'].edge_index = train_edge_index
    train_data = ToUndirected()(train_data)

    # valid data
    valid_data = HeteroData()
    valid_data['track'].node_id = torch.arange(n_track)
    valid_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보
    valid_data['track'].x = x_track
    valid_data['tag'].x = x_tag
    # tag-track matching
    tag_id_mapped = [mapping_tag_index[i] for i in valid_edge['tag_name'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in valid_edge['track_id'].tolist()]
    valid_edge_index = torch.tensor([tag_id_mapped, track_id_mapped], dtype=torch.long)
    valid_data['tag', 'tagged', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
    valid_data = ToUndirected()(valid_data)

    # test data
    test_data = HeteroData()
    test_data['track'].node_id = torch.arange(n_track)
    test_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보 추가
    test_data['track'].x = x_track
    test_data['tag'].x = x_tag
    # tag-track matching
    tag_id_mapped = [mapping_tag_index[i] for i in test_edge['tag_name'].tolist()]
    track_id_mapped = [mapping_track_index[i] for i in test_edge['track_id'].tolist()]
    test_edge_index = torch.tensor([tag_id_mapped, track_id_mapped], dtype=torch.long)
    test_data['tag', 'tagged', 'track'].edge_index = torch.cat([train_edge_index, test_edge_index], dim=1)
    test_data = ToUndirected()(test_data)

    # serving data 저장
    serving_data = HeteroData()
    serving_data['track'].node_id = torch.arange(n_track)
    serving_data['tag'].node_id = torch.arange(n_tag)
    # node 특성 정보 추가
    serving_data['track'].x = x_track
    serving_data['tag'].x = x_tag
    # tag-track matching
    serving_data['tag', 'tagged', 'track'].edge_index = torch.cat([train_edge_index, valid_edge_index, test_edge_index], dim=1)
    serving_data = ToUndirected()(serving_data)
    # graph data 저장
    torch.save(serving_data, f'{args.data_dir}graph_data_{args.model_name}.pt')
    
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index    # valid_edge_index / test_edge_index : train edge를 포함하지 않는 edge 출력


def data_preprocessing(args):
    # 데이터 로드
    tag_track, track_emb, tag_emb = load_data(args)
    
    # Train-Valid-Test 분할
    train_edge, valid_edge, test_edge = train_valid_test_split(tag_track, args.valid_ratio, args.test_ratio)
    
    # 인덱싱
    mapping_track_index, mapping_tag_index = mapping_index(args, tag_track)
    
    # Hetero Graph Data로 변환
    train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index = convert_to_graph(args, train_edge, valid_edge, test_edge,
                                                                                                              track_emb, tag_emb,
                                                                                                              mapping_track_index, mapping_tag_index)
   
    return train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index
