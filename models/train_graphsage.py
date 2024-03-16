import torch
import warnings
from torch_geometric import EdgeIndex
from torch_geometric.loader import LinkNeighborLoader

from graphsage.args import parse_args
from graphsage.data_preprocessing import load_data, train_valid_test_split, mapping_index, convert_to_graph
from graphsage.model import Model, ContrastiveLoss
from graphsage.trainer import train, test
from graphsage.utils import set_seed, makedirs, get_logger


def main():
    args = parse_args()
    set_seed(args.seed)
    makedirs(args.log_dir)
    logger = get_logger(filename=f'{args.log_dir}{args.filename}.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warnings.filterwarnings('ignore')
    
    # 데이터 로드
    interaction, track = load_data(args)
    
    # Train-Valid-Test 분할
    train_interaction, valid_interaction, test_interaction = train_valid_test_split(interaction, args.valid_ratio, args.test_ratio)
    
    # 인덱싱
    mapping_user_index, _, mapping_track_index, _, mapping_artist_index, _ = mapping_index(args, train_interaction, valid_interaction, test_interaction, track)
    
    # Hetero Graph Data로 변환
    train_data, valid_data, test_data, valid_edge_index, test_edge_index = convert_to_graph(args, train_interaction, valid_interaction, test_interaction, track,
                                                                                            mapping_user_index, mapping_track_index, mapping_artist_index)
    
    # Edge 저장
    sparse_size = (train_data['user'].num_nodes, train_data['track'].num_nodes)
    train_edge_index_eval = EdgeIndex(
        train_data['user', 'listen', 'track'].edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    valid_edge_index_eval = EdgeIndex(
        valid_edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    test_edge_index_eval = EdgeIndex(
        test_edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors = {('user', 'listen', 'track'): [args.n_neighbors_sampling] * args.n_layers,    # [node 당 sample 개수] * layer 개수
                        ('track', 'sungby', 'artist'): [args.n_neighbors_sampling] * args.n_layers,
                        ('track', 'rev_listen', 'user'): [args.n_neighbors_sampling] * args.n_layers,
                        ('artist', 'rev_sungby', 'track'): [args.n_neighbors_sampling] * args.n_layers},
        edge_label_index = (('user', 'track'), train_data['user','track'].edge_index),
        neg_sampling = dict(mode='binary', amount=args.negative_sampling),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        persistent_workers = True,
        filter_per_worker=True,
        pin_memory = True,
    )

    model = Model(data=train_data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = ContrastiveLoss(margin=args.margin).to(device)

    counter = 0
    best_val_ndcg = -1
    
    for epoch in range(1, args.epochs+1):
        print(f'Epoch: {epoch:02d}')
        # train
        train_ndcg, train_recall, train_loss = train(model=model, optimizer=optimizer, loss=loss,
                                                     dataloader=train_loader, data=train_data, k=args.k, device=device,
                                                     train_edge_index_eval=train_edge_index_eval)
        logger.info(f'Epoch: {epoch:02d}  Loss: {train_loss:.4f}')
        logger.info(f'Train NDCG@{args.k}: {train_ndcg:.4f}  Train Recall@{args.k}: {train_recall:.4f}')
        
        # validation
        val_ndcg, val_recall = test(model=model, data=valid_data, k=args.k, device=device,
                                    train_edge_index_eval=train_edge_index_eval, test_edge_index_eval=valid_edge_index_eval)
        logger.info(f'Valid NDCG@{args.k}: {val_ndcg:.4f}  Valid Recall@{args.k}: {val_recall:.4f}')
        
        # early stopping
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            logger.info(f'Best Valid NDCG@{args.k} is Updated')
            torch.save(model.state_dict(), f'{args.model_dir}{args.filename}.pt')    # 모델 저장
            counter = 0
        else:
            counter += 1
            if (epoch > args.min_epochs) and (counter >= args.patience):
                logger.info(f'Early stopping at Epoch {epoch:02d}')
                break

    # test
    test_ndcg, test_recall = test(model=model, data=test_data, k=args.k, device=device,
                                  train_edge_index_eval=train_edge_index_eval, test_edge_index_eval=test_edge_index_eval)
    logger.info(f'Test NDCG@{args.k}: {test_ndcg:.4f}   Test Recall@{args.k}: {test_recall:.4f}')


if __name__ == '__main__':
    main()
