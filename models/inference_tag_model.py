import json
import time
import torch
import pandas as pd
from models.graphsage_cbf.args_tag import parse_args
from models.graphsage_cbf.utils import get_logger
from models.graphsage_cbf.model import Model


def inference_tag_model(k, input_tag_list, mapping_index_track, mapping_tag_index, embeddings):
    # input tag 임베딩
    input_tag_list = [mapping_tag_index[str(tag)] for tag in input_tag_list]    # input tag 인덱싱
    input_tag_emb = embeddings['tag'][input_tag_list]    # input tag 임베딩
    
    # track 임베딩
    track_emb = embeddings['track']
    
    # KNN을 통해 tag 임베딩과 track 임베딩 사이의 거리 계산
    # 1. 입력 받은 tag list의 각 tag와 관련된 track 선별
    track_indices = []
    for input_tag_emb_tmp in input_tag_emb:
        scores = torch.matmul(input_tag_emb_tmp, track_emb.T)
        knn_indices = torch.argsort(scores, descending=True)
        knn_indices = knn_indices[:k].tolist()
        track_indices_tmp = knn_indices    # knn index와 track index 동일
        track_indices.extend(track_indices_tmp)
    track_indices = list(set(track_indices))
    
    # 2. 입력 받은 tag list의 종합 점수로 ranking
    if len(input_tag_list) > 1:
        selected_track_emb = embeddings['track'][track_indices]    # 선별한 각 tag 별 track만 사용
        
        scores = torch.matmul(input_tag_emb, selected_track_emb.T)
        total_scores = scores.sum(axis=0)    # 종합 거리 점수 계산
        knn_indices = torch.argsort(total_scores, descending=True)
        knn_indices = knn_indices[:k].tolist()
        track_indices = [track_indices[i] for i in knn_indices]    # knn index를 track index로 변환
    
    recommend_track_list = [mapping_index_track[str(i)] for i in track_indices]   # track index를 track_id로 변환
    
    return recommend_track_list


def load_data_for_tag_model():
    args = parse_args()    # 파라미터 로드
    
    # track index mapping 정보 로드
    with open(f'{args.data_dir}mapping_{args.model_name}_index_track.json', 'r') as f:
        mapping_index_track = json.load(f)
    with open(f'{args.data_dir}mapping_{args.model_name}_tag_index.json', 'r') as f:
        mapping_tag_index = json.load(f)
    
    # 그래프 데이터 로드
    graph_data = torch.load(f'{args.data_dir}{args.graph_filename}')

    return mapping_index_track, mapping_tag_index, graph_data


def load_tag_model(graph_data):
    args = parse_args()    # 파라미터 로드
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
    
    # 입력 받은 tag list
    input_tag_list = args.input_tag_list
    
    # 데이터 로드
    track = pd.read_csv(f'{args.data_dir}{args.track_filename}', usecols=['track_id','track_name','artist_name','tag_name_list'])
    track = track[['track_id','track_name','artist_name','tag_name_list']]
    mapping_index_track, mapping_tag_index, graph_data = load_data_for_tag_model()
    
    # 데이터 소요 시간
    data_time = time.time()
    elapsed_time = data_time - start_time
    print(f'Data Loading : {elapsed_time:.2f}s')
    
    # 모델 로드 & 임베딩
    embeddings = load_tag_model(graph_data)
    
    # 모델 소요 시간
    model_time = time.time()
    elapsed_time = model_time - data_time
    print(f'Model Loading : {elapsed_time:.2f}s')
    
    # 추천
    recommend_track_list = inference_tag_model(args.k, input_tag_list, mapping_index_track, mapping_tag_index, embeddings)
    
    # 추천 소요 시간
    rec_time = time.time()
    elapsed_time = rec_time - model_time
    print(f'Recommending : {elapsed_time:.2f}s')
    
    # 전체 소요 시간
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total Inference Time : {elapsed_time:.2f}s')
    
    # 추천된 track 정보 출력
    logger.info('Recommended Tracks')
    recommend_info = pd.merge(pd.DataFrame(recommend_track_list, columns=['track_id']), track, how='left', on='track_id')
    logger.info('\n' + recommend_info.to_string())

if __name__ == '__main__':
    main()
