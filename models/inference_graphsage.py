import json
import time
import pandas as pd
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import MIPSKNNIndex
from models.graphsage.args import parse_args
from models.graphsage.utils import get_logger
from models.graphsage.model import Model


def load_user_data(args):
    # 유저 정보 로드
    input_data = pd.read_csv(f'{args.data_dir}{args.serving_filename}', index_col=0)

    return input_data


def load_meta_data(args):
    # track 정보
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}', usecols=['track_id','track_name','artist_name','tags'])
    track = track[['track_id','track_name','artist_name','tags']]

    # user/track/artist index mapping 정보 로드
    with open(f'{args.data_dir}mapping_track_index.json', 'r') as f:
        mapping_track_index = json.load(f)
    with open(f'{args.data_dir}mapping_index_track.json', 'r') as f:
        mapping_index_track = json.load(f)

    return track, mapping_track_index, mapping_index_track


def load_graph_data(args, input_data, mapping_track_index):
    # 기존 그래프 데이터 로드
    data = torch.load(f'{args.data_dir}{args.graph_filename}')

    # user-track interaction 추가
    new_user_id = int(data['user']['node_id'][-1])    # 새로운 user의 인덱스를 데이터 내의 마지막 인덱스로 배정
    user_id_mapped = [new_user_id for _ in input_data['user_id'].tolist()]    # 새로운 interaction의 user 인덱스
    track_id_mapped = [mapping_track_index[str(i)] for i in input_data['track_id'].tolist()]    # 새로운 interaction의 track 인덱스
    data_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)    # 새로운 interaction의 user, track 조합
    data['user', 'listen', 'track'].edge_index = torch.cat([data['user','listen','track'].edge_index, data_edge_index], dim=1)    # 기존 interaction에 새로운 interaction 추가

    data = ToUndirected()(data)    # interaction을 양방향으로 설정

    return data, new_user_id


def inference_graphsage(input_data, k=20):
    # 기본 설정
    start_time = time.time()    # inference 소요 시간 계산
    args = parse_args()    # 파라미터 로드
    logger = get_logger(filename=f'{args.log_dir}inference_{args.log_filename}')    # 로그 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # GPU 설정
    
    # 데이터 로드
    track, mapping_track_index, mapping_index_track = load_meta_data(args)
    data, new_user_id = load_graph_data(args, input_data, mapping_track_index)    # 그래프 데이터 로드

    # 데이터 소요 시간
    data_time = time.time()
    elapsed_time = data_time - start_time
    print(f'Data Loading : {elapsed_time:.2f}s')
    
    # 모델 로드
    model = Model(data=data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f'{args.model_dir}{args.model_filename}'))
    model.eval()    # 모델 평가 모드

    # 모델 로드 소요 시간
    model_load_time = time.time()
    elapsed_time = model_load_time - data_time
    print(f'Model Loading : {elapsed_time:.2f}s')
    
    # 모델을 통해 임베딩 추출
    embeddings = model(data.to(device))
    user_emb = embeddings['user']
    track_emb = embeddings['track']
    # tag_emb = embeddings['tag']    # input tag 기반 track 추천 추가
    new_user_emb = user_emb[new_user_id].unsqueeze(0)    # 대상 user 임베딩 추출

    # 모델 추론 소요 시간
    model_inference_time = time.time()
    elapsed_time = model_inference_time - model_load_time
    print(f'Model Running : {elapsed_time:.2f}s')
    
    # KNN을 통해 user 임베딩과 track 임베딩 사이의 거리 계산
    # MIPS(maximum inner product search) 기반 KNN
    mips = MIPSKNNIndex(track_emb)
    _, track_indices = mips.search(new_user_emb, k)    # 대상 user와 가까운 track 출력
    recommend_list = track_indices.tolist()[0]
    recommend_list = [mapping_index_track[str(i)] for i in recommend_list]   # 인덱싱된 track을 track_id로 변환
    
    # 추천할 user 입력 데이터의 track 정보 출력
    logger.info('Input Tracks')
    input_data_info = pd.merge(pd.DataFrame(input_data, columns=['track_id']), track, how='left',on='track_id')
    logger.info('\n' + input_data_info.to_string())
    
    # 추천된 track 정보 출력
    # logger.info('Recommended Tracks')
    recommend_info = pd.merge(pd.DataFrame(recommend_list, columns=['track_id']), track, how='left',on='track_id')
    # logger.info('\n' + recommend_info.to_string())
    
    # 추천된 track 중 입력 데이터에 있는 track 정보 출력
    logger.info('Recommended Input Tracks')
    input_recommend_info = recommend_info[recommend_info['track_id'].isin(input_data_info['track_id'])]
    logger.info('\n' + input_recommend_info.to_string())
    
    # 추천된 track 중 입력 데이터에 없는 track 정보 출력
    logger.info('Recommended New Tracks')
    new_recommend_info = recommend_info[~recommend_info['track_id'].isin(input_data_info['track_id'])]
    logger.info('\n' + new_recommend_info.to_string())

    # 추천 소요 시간
    rec_time = time.time()
    elapsed_time = rec_time - model_inference_time
    print(f'Recommending : {elapsed_time:.2f}s')
    
    # 전체 소요 시간
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total Inference Time : {elapsed_time:.2f}s')
    
    return recommend_list[:k]


def main():
    args = parse_args()
    input_data = load_user_data(args)
    _ = inference_graphsage(input_data, k=args.k)


if __name__ == '__main__':
    main()