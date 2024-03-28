import json
import time
import torch
import pandas as pd
from torch_geometric.transforms import ToUndirected
from models.graphsage_cf.args import parse_args
from models.graphsage_cf.utils import get_logger
from models.graphsage_cf.model import Model


def inference_cf_model(k, candidate_track_list, mapping_index_track, mapping_track_index, embeddings, new_user_id):
    # user 임베딩
    new_user_emb = embeddings['user'][new_user_id]    # 대상 user 임베딩 추출
    
    # candidate track 임베딩
    candidate_track_index_list = [mapping_track_index[str(track)] for track in candidate_track_list]    # candidate track 인덱싱
    candidate_track_emb = embeddings['track'][candidate_track_index_list]    # candidate track 임베딩
    
    # KNN을 통해 user 임베딩과 track 임베딩 사이의 거리 계산
    scores = torch.matmul(new_user_emb, candidate_track_emb.T)
    knn_indices = torch.argsort(scores, descending=True)
    knn_indices = knn_indices[:k].tolist()
    track_indices = [candidate_track_index_list[i] for i in knn_indices]    # knn index를 track index로 변환
    
    recommend_track_list = [mapping_index_track[str(i)] for i in track_indices]   # 인덱싱된 track을 track_id로 변환
    
    return recommend_track_list


def load_data_for_cf_model(args, input_track_list):
    # track 정보
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}', usecols=['track_id','track_name','artist_name','tag_name_list','track_emb'])

    # 입력 받은 track list의 임베딩 정보
    track_emb = pd.merge(pd.DataFrame(input_track_list, columns=['track_id']), track[['track_id','track_emb']], how='left', on='track_id')
    
    # 필요한 track 정보만 추출
    track = track[['track_id','track_name','artist_name','tag_name_list']]
    
    # track index mapping 정보 로드
    with open(f'{args.data_dir}mapping_{args.model_name}_index_track.json', 'r') as f:
        mapping_index_track = json.load(f)
    with open(f'{args.data_dir}mapping_{args.model_name}_track_index.json', 'r') as f:
        mapping_track_index = json.load(f)
    
    # 그래프 데이터 로드
    graph_data = torch.load(f'{args.data_dir}{args.graph_filename}')
    
    return track, track_emb, mapping_index_track, mapping_track_index, graph_data


def update_user_graph_data(graph_data, input_track_list, track_emb, mapping_track_index):
    del graph_data['track', 'rev_listen', 'user'], graph_data['track', 'rev_tagged', 'tag']    # 기존 그래프 데이터의 reverse edge 제거
    
    # 새로운 user 임베딩 생성
    track_emb['track_emb'] = track_emb['track_emb'].apply(eval)    # read_csv로 받은 텍스트 형태의 track_emb를 리스트로 변경
    new_user_emb = [round(sum(x) / len(x), 4) for x in zip(*list(track_emb['track_emb']))]    # track_emb 평균 계산
    graph_data['user'].x[-1] = torch.tensor(new_user_emb)
    
    # user-track interaction 추가
    new_user_id = int(graph_data['user']['node_id'][-1])    # 새로운 user의 인덱스를 데이터 내의 마지막 인덱스로 배정
    user_id_mapped = [new_user_id for _ in range(len(input_track_list))]    # 새로운 interaction의 user 인덱스
    track_id_mapped = [mapping_track_index[str(i)] for i in input_track_list]    # 새로운 interaction의 track 인덱스
    data_edge_index = torch.tensor([user_id_mapped, track_id_mapped], dtype=torch.long)    # 새로운 interaction의 user, track 조합
    graph_data['user', 'listen', 'track'].edge_index = torch.cat([graph_data['user','listen','track'].edge_index, data_edge_index], dim=1)    # 기존 interaction에 새로운 interaction 추가
    
    graph_data = ToUndirected()(graph_data)    # interaction을 양방향으로 설정
    
    return graph_data, new_user_id


def load_cf_model(args, graph_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # GPU 설정
    
    # 모델 로드
    model = Model(data=graph_data, x_dim=args.x_dim, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f'{args.model_dir}{args.model_filename}'))
    model.eval()    # 모델 평가 모드
    
    # 모델을 통해 임베딩 추출
    embeddings = model(graph_data.to(device))
    
    return embeddings


def main():
    # 기본 설정
    start_time = time.time()    # inference 소요 시간 계산
    args = parse_args()    # 파라미터 로드
    logger = get_logger(filename=f'{args.log_dir}inference_{args.model_name}.log')    # 로그 설정
    
    # 입력 받은 track list
    input_data = pd.read_csv(f'{args.data_dir}{args.input_track_list}', index_col=0)
    input_track_list = input_data['track_id'][input_data['track_id'] < 49707345].to_list()
    
    # 데이터 로드
    track, track_emb, mapping_index_track, mapping_track_index, graph_data = load_data_for_cf_model(args, input_track_list)
    
    # 그래프 데이터 생성
    updated_graph_data, new_user_id = update_user_graph_data(graph_data, input_track_list, track_emb, mapping_track_index)
    
    # 데이터 소요 시간
    data_time = time.time()
    elapsed_time = data_time - start_time
    print(f'Data Loading : {elapsed_time:.2f}s')
    
    # 모델 로드 & 임베딩
    embeddings = load_cf_model(args, updated_graph_data)
    
    # 모델 소요 시간
    model_time = time.time()
    elapsed_time = model_time - data_time
    print(f'Model Loading : {elapsed_time:.2f}s')
    
    # 추천
    candidate_track_list = [*mapping_track_index.keys()]    # 모든 track을 후보로 설정
    recommend_track_list = inference_cf_model(args.k, candidate_track_list, mapping_index_track, mapping_track_index, embeddings, new_user_id)
    
    # 추천 소요 시간
    rec_time = time.time()
    elapsed_time = rec_time - model_time
    print(f'Recommending : {elapsed_time:.2f}s')
    
    # 전체 소요 시간
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total Inference Time : {elapsed_time:.2f}s')
    
    # 입력된 track 정보 출력
    logger.info('Recommended Tracks')
    recommend_info = pd.merge(pd.DataFrame(input_track_list, columns=['track_id']), track, how='left', on='track_id')
    logger.info('\n' + recommend_info.to_string())
    
    # 추천된 track 정보 출력
    logger.info('Recommended Tracks')
    recommend_info = pd.merge(pd.DataFrame(recommend_track_list, columns=['track_id']), track, how='left', on='track_id')
    logger.info('\n' + recommend_info.to_string())

if __name__ == '__main__':
    main()
