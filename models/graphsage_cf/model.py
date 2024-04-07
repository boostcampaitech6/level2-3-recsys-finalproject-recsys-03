import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class GraphSAGE(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SAGEConv((-1, -1), hidden_dim, 'mean'))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:    # 마지막 layer를 제외하고 ReLU 적용
                x = x.relu()
        return x


class Model(torch.nn.Module):
    def __init__(self, data, x_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.data = data
        self.encoder = GraphSAGE(hidden_dim, n_layers)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')

        # input 초기값 설정 : 임베딩 + 특성 linear (특성 정보가 없는 경우, 임베딩만 사용)
        if hasattr(data['track'], 'x'):
            self.user_emb = nn.Embedding(num_embeddings=data['user'].num_nodes, embedding_dim=int(embedding_dim/2))
            self.track_emb = nn.Embedding(num_embeddings=data['track'].num_nodes, embedding_dim=int(embedding_dim/2))
            self.tag_emb = nn.Embedding(num_embeddings=data['tag'].num_nodes, embedding_dim=int(embedding_dim/2))
            
            self.user_feature_transform = nn.Linear(x_dim, int(embedding_dim/2))
            self.track_feature_transform = nn.Linear(x_dim, int(embedding_dim/2))
            self.tag_feature_transform = nn.Linear(x_dim, int(embedding_dim/2))
        
        else:
            self.user_emb = nn.Embedding(num_embeddings=data['user'].num_nodes, embedding_dim=embedding_dim)
            self.track_emb = nn.Embedding(num_embeddings=data['track'].num_nodes, embedding_dim=embedding_dim)
            self.tag_emb = nn.Embedding(num_embeddings=data['tag'].num_nodes, embedding_dim=embedding_dim)

    def forward(self, data):
        # 학습된 임베딩 가져오기
        user_emb = self.user_emb(data['user'].node_id)
        track_emb = self.track_emb(data['track'].node_id)
        tag_emb = self.tag_emb(data['tag'].node_id)
        
        if hasattr(data['track'], 'x'):
            user_features = F.relu(self.user_feature_transform(data['user'].x))
            track_features = F.relu(self.track_feature_transform(data['track'].x))
            tag_features = F.relu(self.tag_feature_transform(data['tag'].x))
            
            user_emb = torch.cat([user_emb, user_features], dim=1)
            track_emb = torch.cat([track_emb, track_features], dim=1)
            tag_emb = torch.cat([tag_emb, tag_features], dim=1)
        
        # # 임베딩 딕셔너리
        x_dict = {'user': user_emb,
                  'track': track_emb,
                  'tag': tag_emb}
        
        # GraphSAGE 통과
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        return x_dict
