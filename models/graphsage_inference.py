import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


def graphsage_load_data():
    # 유저 정보 로드
    input_data = pd.read_csv('./data/input_data.csv', index_col=0)

    # track index mapping 정보 로드
    with open('./data/mapping_track_index.json', 'r') as f:
        mapping_track_index = json.load(f)
    with open('./data/mapping_index_track.json', 'r') as f:
        mapping_index_track = json.load(f)

    return input_data, mapping_track_index, mapping_index_track


def convert_to_graph(input_data, mapping_track_index):
    # 그래프 데이터로 변환
    data = HeteroData()
    data['user'].node_id = torch.arange(1)
    data['track'].node_id = torch.arange(len(mapping_track_index))
    user_ids_mapped = [0 for _ in list(input_data['user_id'])]
    track_ids_mapped = [mapping_track_index[str(u)] for u in list(input_data['track_id'])]
    data['user', 'listen', 'track'].edge_index = torch.tensor([user_ids_mapped, track_ids_mapped], dtype=torch.long)
    data = T.ToUndirected()(data)
    
    return data


# 모델 정의
class GraphSAGE(nn.Module):
    def __init__(self, input_feats, hidden_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, 'mean')

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

class DotPredictor(nn.Module):
    def forward(self, h_user, h_track, edge_index):
        return (h_user[edge_index[0]] * h_track[edge_index[1]]).sum(dim=1)

class Model(torch.nn.Module):
    def __init__(self, data, input_size=64, hidden_size=64):
        super().__init__()
        self.data = data
        self.user_emb = torch.nn.Embedding(self.data['user'].num_nodes, hidden_size)
        self.track_emb = torch.nn.Embedding(self.data['track'].num_nodes, hidden_size)

        self.gnn = GraphSAGE(input_size, hidden_size)
        self.gnn = to_hetero(self.gnn, metadata=self.data.metadata())
        self.dotpredictor = DotPredictor()

    def forward(self, data: HeteroData):
        x_dict = {
          'user': self.user_emb(data['user'].node_id),
          'track': self.track_emb(data['track'].node_id),
        } 
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred_pos_score = self.dotpredictor(
            x_dict['user'],
            x_dict['track'],
            data['user', "listen", 'track'].pos_edge_label_index,
        )
        pred_neg_score = self.dotpredictor(
            x_dict['user'],
            x_dict['track'],
            data['user', "listen", 'track'].neg_edge_label_index,
        )
        return pred_pos_score, pred_neg_score
    
    def encode_user_track(self, data: HeteroData):
        x_dict = {
          'user': self.user_emb(data['user'].node_id),
          'track': self.track_emb(data['track'].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict['user'], x_dict['track']


def graphsage_inference(k):
    # 데이터 로드
    input_data, mapping_track_index, mapping_index_track = graphsage_load_data()
    
    # 그래프 데이터로 변환
    data = convert_to_graph(input_data, mapping_track_index)
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = torch.load('./model/graphsage.pt')


    # (user * track) score 계산
    user_embedded, track_embedded = model.encode_user_track(data.to(device))
    score = (user_embedded * track_embedded).sum(dim=1)

    # score 순으로 정렬 후 추천 track_id 출력
    score = score.tolist()
    score = [(i, s) for i, s in enumerate(score)]
    score.sort(key=lambda x: -x[1])
    recommend = [mapping_index_track[str(i)] for i, _ in score]

    return recommend[:k]
