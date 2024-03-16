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
    def __init__(self, data, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.data = data
        self.encoder = GraphSAGE(hidden_dim, n_layers)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')

        self.user_emb = nn.Embedding(num_embeddings=data['user'].num_nodes, embedding_dim=emb_dim)
        self.track_emb = nn.Embedding(num_embeddings=data['track'].num_nodes, embedding_dim=emb_dim)
        self.artist_emb = nn.Embedding(num_embeddings=data['artist'].num_nodes, embedding_dim=emb_dim)
        self.tag_emb = nn.Embedding(num_embeddings=data['tag'].num_nodes, embedding_dim=emb_dim)

    def forward(self, data):
        # 학습된 임베딩 가져오기
        x_dict = {'user': self.user_emb(data['user'].node_id),
                  'track': self.track_emb(data['track'].node_id),
                  'artist': self.artist_emb(data['artist'].node_id),
                  'tag': self.tag_emb(data['tag'].node_id)}
        # GraphSAGE 통과
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        return x_dict


# Contrastive Loss : positive 쌍의 임베딩은 서로 가깝게, negative 쌍의 임베딩은 서로 멀어지게 학습
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin    # margin : positive와 negative 쌍 사이의 최소 거리

    def forward(self, user_emb, track_emb):
        # Euclidean distance 계산
        pos_neg_split = user_emb.size(0)//2    # embedding은 pos_edge와 neg_edge가 1:1 비율로 존재
        pos_distance = F.pairwise_distance(user_emb[:pos_neg_split], track_emb[:pos_neg_split], keepdim=True)
        neg_distance = F.pairwise_distance(user_emb[pos_neg_split:], track_emb[pos_neg_split:], keepdim=True)
        losses = torch.relu(self.margin + pos_distance - neg_distance)
        return losses.mean()
