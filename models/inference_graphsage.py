import json
import time
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero, MIPSKNNIndex
from graphsage.args import parse_args
from graphsage.utils import get_logger


def load_data(args):
    # 유저 정보 로드
    input_data = pd.read_csv(f'{args.data_dir}{args.serving_interaction_filename}', index_col=0)
    
    # track 정보
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}', usecols=['track_id','uri','track_name','artist_name','tags'])
    track = track[['track_id','uri','track_name','artist_name','tags']]

    # user/track/artist index mapping 정보 로드
    with open(f'{args.data_dir}mapping_track_index.json', 'r') as f:
        mapping_track_index = json.load(f)
    with open(f'{args.data_dir}mapping_index_track.json', 'r') as f:
        mapping_index_track = json.load(f)

    return input_data, track, mapping_track_index, mapping_index_track


def load_graph_data(args, input_data, mapping_track_index):
    # 기존 그래프 데이터 로드
    data = torch.load(f'{args.data_dir}serving_data_{args.filename}.pt')

    # user-track interaction 추가
    new_user_id = int(data['user']['node_id'][-1])    # 새로운 user의 인덱스를 데이터 내의 마지막 인덱스로 배정
    user_id_mapped = [new_user_id for _ in input_data['user_id'].tolist()]    # 새로운 interaction의 user 인덱스
    track_id_mapped = [mapping_track_index[str(i)] for i in input_data['track_id'].tolist()]    # 새로운 interaction의 track 인덱스
    data_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)    # 새로운 interaction의 user, track 조합
    data['user', 'listen', 'track'].edge_index = torch.cat([data['user','listen','track'].edge_index, data_edge_index], dim=1)    # 기존 interaction에 새로운 interaction 추가

    data = ToUndirected()(data)    # interaction을 양방향으로 설정

    return data, new_user_id


# GraphSAGE 모델 구조
class GraphSAGE(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SAGEConv((-1, -1), hidden_dim, 'mean'))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1: # 마지막 레이어가 아니라면 ReLU 적용
                x = x.relu()
        return x


# 모델 정의
class Model(torch.nn.Module):
    def __init__(self, data, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.data = data
        # self.encoder = GraphSAGE(hidden_dim)
        self.encoder = GraphSAGE(hidden_dim, n_layers)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')

        self.user_emb = nn.Embedding(num_embeddings=data['user'].num_nodes, embedding_dim=emb_dim)
        self.track_emb = nn.Embedding(num_embeddings=data['track'].num_nodes, embedding_dim=emb_dim)
        self.artist_emb = nn.Embedding(num_embeddings=data['artist'].num_nodes, embedding_dim=emb_dim)

    def forward(self, data):
        # 학습된 임베딩 가져오기
        x_dict = {'user': self.user_emb(data['user'].node_id),
                  'track': self.track_emb(data['track'].node_id),
                  'artist': self.artist_emb(data['artist'].node_id)}
        # GraphSAGE 통과
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        return x_dict


def inference():
    # 기본 설정
    start_time = time.time()    # inference 소요 시간 계산
    args = parse_args()    # 파라미터 로드
    logger = get_logger(filename=f'{args.log_dir}{args.filename}_inference.log')    # 로그 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # GPU 설정
    
    # 데이터 로드
    input_data, track, mapping_track_index, mapping_index_track = load_data(args)
    data, new_user_id = load_graph_data(args, input_data, mapping_track_index)    # 그래프 데이터 로드

    # 데이터 소요 시간
    data_time = time.time()
    elapsed_time = data_time - start_time
    print(f'Data Time : {elapsed_time:.2f}s')
    
    # 모델 로드
    model = Model(data=data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f'{args.model_dir}{args.filename}.pt'))
    model.eval()    # 모델 평가 모드

    # 모델 로드 소요 시간
    model_load_time = time.time()
    elapsed_time = model_load_time - data_time
    print(f'Model Load Time : {elapsed_time:.2f}s')
    
    # 모델을 통해 임베딩 추출
    embeddings = model(data.to(device))
    user_emb = embeddings['user']
    track_emb = embeddings['track']
    new_user_emb = user_emb[new_user_id].unsqueeze(0)    # 대상 user 임베딩 추출

    # 모델 추론 소요 시간
    model_inference_time = time.time()
    elapsed_time = model_inference_time - data_time
    print(f'Model Run Time : {elapsed_time:.2f}s')
    
    # KNN을 통해 user 임베딩과 track 임베딩 사이의 거리 계산
    # MIPS(maximum inner product search) 기반 KNN
    mips = MIPSKNNIndex(track_emb)
    _, track_indices = mips.search(new_user_emb, args.k)    # 대상 user와 가까운 track 출력
    recommend_list = track_indices.tolist()[0]
    recommend_list = [mapping_index_track[str(i)] for i in recommend_list]   # 인덱싱된 track을 track_id로 변환
    
    # 추천할 user가 선호하는 track 정보 출력
    logger.info('Input Tracks')
    input_data_info = pd.merge(pd.DataFrame(input_data, columns=['track_id']), track, how='left',on='track_id')
    logger.info('\n'+input_data_info.to_string())
    
    # 추천된 track 정보 출력
    logger.info('Recommended Tracks')
    recommend_info = pd.merge(pd.DataFrame(recommend_list, columns=['track_id']), track, how='left',on='track_id')
    logger.info('\n'+recommend_info.to_string())
    
    # 추천된 track 중 입력 track에 있는 track 정보 출력
    logger.info('Recommended Old Tracks')
    old_recommend_info = recommend_info[recommend_info['track_id'].isin(input_data_info['track_id'])]
    logger.info('\n'+old_recommend_info.to_string())
    
    # 추천된 track 중 입력 track에 없는 track 정보 출력
    logger.info('Recommended New Tracks')
    new_recommend_info = recommend_info[~recommend_info['track_id'].isin(input_data_info['track_id'])]
    logger.info('\n'+new_recommend_info.to_string())

    # 추천 소요 시간
    rec_time = time.time()
    elapsed_time = rec_time - model_inference_time
    print(f'Rec Time : {elapsed_time:.2f}s')
    
    # 전체 소요 시간
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total Inference Time : {elapsed_time:.2f}s')
    
    return recommend_list[:args.k]


def main():
    _ = inference()


if __name__ == '__main__':
    main()
