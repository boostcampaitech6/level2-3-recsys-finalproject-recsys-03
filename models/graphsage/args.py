import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 데이터 파라미터
    parser.add_argument('--data_dir', default='../data/', type=str, help='data 폴더 경로')
    parser.add_argument('--log_dir', default='./log/', type=str, help='log 폴더 경로')
    parser.add_argument('--model_dir', default='./model/', type=str, help='model 폴더 경로')
    parser.add_argument('--user_filename', default='user_emb.csv', type=str, help='user 파일명 (csv 형식)')
    parser.add_argument('--track_filename', default='track_emb.csv', type=str, help='track 파일명 (csv 형식)')
    parser.add_argument('--artist_filename', default='artist_emb.csv', type=str, help='artist 파일명 (csv 형식)')
    parser.add_argument('--tag_filename', default='tag_emb.csv', type=str, help='tag 파일명 (csv 형식)')
    parser.add_argument('--filename', default='graphsage', type=str, help='생성되는 파일명 (형식 자동 지정)')

    # 모델 파라미터
    parser.add_argument('--embedding_dim', default=64, type=int, help='GraphSAGE의 노드 임베딩 차원')
    parser.add_argument('--hidden_dim', default=64, type=int, help='GraphSAGE의 hidden layer 차원')
    parser.add_argument('--n_layers', default=2, type=int, help='GraphSAGE의 SAGEConv 개수')
    parser.add_argument('--neighbors_sampling', default=30, type=int, help='Graph Message Passing 과정에서 이웃 노드 Sampling 개수')
    parser.add_argument('--negative_sampling', default=1, type=int, help='Negative Sampling 비율 (Positive interaction의 n배)')
    
    # 학습 파라미터
    parser.add_argument('--seed', default=24, type=int, help='seed')
    parser.add_argument("--valid_ratio", default=0.1, type=float, help='Validation 데이터 비율 (user 기준으로 분할)')
    parser.add_argument("--test_ratio", default=0.1, type=float, help='Test 데이터 비율 (user 기준으로 분할)')
    parser.add_argument('--batch_size', default=1024, type=int, help='DataLoader batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning Rate')
    parser.add_argument('--margin', default=1, type=int, help='Contrastive Loss의 margin')
    parser.add_argument('--epochs', default=50, type=int, help='Epoch 수')
    parser.add_argument('--min_epochs', default=10, type=int, help='최소 Epoch 수')
    parser.add_argument('--early_stopping', default=10, type=int, help='Validation 성능이 특정 횟수 동안 향상되지 않았을 때 Early Stopping 실행')
    parser.add_argument('--topk', default=100, type=int, help='Top@k 평가 지표: 추천 track 개수')

    # 추론 파라미터
    # 추론 과정에서 사용되는 파라미터 : serving_filename, k, data_dir, model_dir, track_filename, filename, embedding_dim, hidden_dim, n_layers
    # 파라미터 중 serving_filename과 k만 변경 가능, 다른 파라미터들은 학습 상황과 동일하게 입력
    parser.add_argument('--serving_filename', default='input_data.csv', type=str, help='interaction 파일명 (csv 형식)')
    parser.add_argument('--k', default=100, type=int, help='추천 track 개수')

    args = parser.parse_args()

    return args
