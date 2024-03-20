import torch
import warnings
from torch_geometric import EdgeIndex
from torch_geometric.loader import LinkNeighborLoader

from graphsage.args import parse_args
from graphsage.data_preprocessing import data_preprocessing
from graphsage.model import Model, ContrastiveLoss
from graphsage.trainer import train, test
from graphsage.utils import set_seed, makedirs, get_logger


def main():
    args = parse_args()
    set_seed(args.seed)
    makedirs(args.log_dir)
    logger = get_logger(filename=f'{args.log_dir}{args.log_filename}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warnings.filterwarnings('ignore')
    
    # 데이터 전처리
    train_data, valid_data, test_data, train_edge_index, valid_edge_index, test_edge_index = data_preprocessing(args)
    
    # Edge 저장
    sparse_size = (train_data['user'].num_nodes, train_data['track'].num_nodes)
    train_edge_index = EdgeIndex(
        train_edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    valid_edge_index = EdgeIndex(
        valid_edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    test_edge_index = EdgeIndex(
        test_edge_index.to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]

    # DataLoader
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors = {('user', 'listen', 'track'): [args.neighbors_sampling] * args.n_layers,    # [node 당 sample 개수] * layer 개수
                        ('track', 'sungby', 'artist'): [args.neighbors_sampling] * args.n_layers,
                        ('track', 'tagged', 'tag'): [args.neighbors_sampling] * args.n_layers,
                        ('track', 'rev_listen', 'user'): [args.neighbors_sampling] * args.n_layers,
                        ('artist', 'rev_sungby', 'track'): [args.neighbors_sampling] * args.n_layers,
                        ('tag', 'rev_tagged', 'track'): [args.neighbors_sampling] * args.n_layers},
        edge_label_index = (('user', 'listen', 'track'), train_data['user', 'listen', 'track'].edge_index),
        # neg_sampling = dict(mode='binary', amount=args.negative_sampling),
        batch_size = args.batch_size,
        shuffle = True,
        # num_workers = 4,
        # persistent_workers = True,
        filter_per_worker=True,
        # pin_memory = True,
    )

    # 모델 정의
    model = Model(data=train_data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = ContrastiveLoss(margin=args.margin, negative_sampling=args.negative_sampling).to(device)

    counter_interaction = 0
    counter_content = 0
    best_val_interaction_dict = {'epoch':0, 'train_loss':0, 'train_ndcg':0, 'train_recall':0, 'val_ndcg':0, 'val_recall':0}
    best_val_content_dict = {'epoch':0, 'train_loss':0, 'train_ndcg':0, 'train_recall':0, 'val_ndcg':0, 'val_recall':0}
    for epoch in range(1, args.epochs+1):
        print(f'Epoch: {epoch:02d}')
        # train
        train_ndcg, train_recall, train_loss = train(args, model=model, optimizer=optimizer, loss=loss,
                                                     dataloader=train_loader, data=train_data, train_edge_index=train_edge_index,
                                                     k=args.topk, device=device)
        logger.info(f'Epoch: {epoch:02d}  Loss: {train_loss:.4f}')
        logger.info(f'Train Interaction NDCG@{args.topk}: {train_ndcg[0]:.4f}  Train Interaction Recall@{args.topk}: {train_recall[0]:.4f}')
        logger.info(f'Train   Content   NDCG@{args.topk}: {train_ndcg[1]:.4f}  Train   Content   Recall@{args.topk}: {train_recall[1]:.4f}')
        
        # validation
        val_ndcg, val_recall = test(model=model, data=valid_data, k=args.topk, device=device, train_edge_index=train_edge_index, test_edge_index=valid_edge_index)
        logger.info(f'Valid Interaction NDCG@{args.topk}: {val_ndcg[0]:.4f}  Valid Interaction Recall@{args.topk}: {val_recall[0]:.4f}')
        logger.info(f'Valid   Content   NDCG@{args.topk}: {val_ndcg[1]:.4f}  Valid   Content   Recall@{args.topk}: {val_recall[1]:.4f}')
        
        # Best 모델 저장
        if val_ndcg[0] > best_val_interaction_dict['train_ndcg']:
            logger.info(f'Best Valid Interaction NDCG@{args.topk} is Updated')
            best_val_interaction_dict = {'epoch': epoch, 'train_loss': train_loss, 'train_ndcg': train_ndcg[0], 'train_recall': train_recall[0], 'val_ndcg': val_ndcg[0], 'val_recall': val_recall[0]}
            torch.save(model.state_dict(), f'{args.model_dir}interaction_{args.model_filename}')    # interaction 모델 저장
            counter_interaction = 0
        else:
            counter_interaction += 1
        
        if val_ndcg[1] > best_val_content_dict['train_ndcg']:
            logger.info(f'Best Valid Content NDCG@{args.topk} is Updated')
            best_val_content_dict = {'epoch': epoch, 'train_loss': train_loss, 'train_ndcg': train_ndcg[1], 'train_recall': train_recall[1], 'val_ndcg': val_ndcg[1], 'val_recall': val_recall[1]}
            torch.save(model.state_dict(), f'{args.model_dir}content_{args.model_filename}')    # content 모델 저장
            counter_content = 0
        else:
            counter_content += 1
        
        # early stopping (최소 epoch 이상이면서 지정된 횟수의 epoch 동안 성능 향상 없을 때)
        if (epoch > args.min_epochs) and (counter_interaction >= args.early_stopping) and (counter_content >= args.early_stopping):
            logger.info(f'Early stopping at Epoch {epoch:02d}')
            break
    
    # 마지막 epoch 모델 저장
    torch.save(model.state_dict(), f'{args.model_dir}{args.model_filename}')
    
    # Best 모델 결과 출력
    logger.info('Best Interaction Model')
    logger.info(f'Epoch: {best_val_interaction_dict["epoch"]:02d}  Loss: {best_val_interaction_dict["train_loss"]:.4f}')
    logger.info(f'Train Interaction NDCG@{args.topk}: {best_val_interaction_dict["train_ndcg"]:.4f}  Train Interaction Recall@{args.topk}: {best_val_interaction_dict["train_recall"]:.4f}')
    logger.info(f'Valid Interaction NDCG@{args.topk}: {best_val_interaction_dict["val_ndcg"]:.4f}  Valid Interaction Recall@{args.topk}: {best_val_interaction_dict["val_recall"]:.4f}')
    
    logger.info('Best Content Model')
    logger.info(f'Epoch: {best_val_content_dict["epoch"]:02d}  Loss: {best_val_content_dict["train_loss"]:.4f}')
    logger.info(f'Train   Content   NDCG@{args.topk}: {best_val_content_dict["train_ndcg"]:.4f}  Train   Content   Recall@{args.topk}: {best_val_content_dict["train_recall"]:.4f}')
    logger.info(f'Valid   Content   NDCG@{args.topk}: {best_val_content_dict["val_ndcg"]:.4f}  Valid   Content   Recall@{args.topk}: {best_val_content_dict["val_recall"]:.4f}')
    
    # test
    logger.info('Test with Last Epoch Model')
    test_ndcg, test_recall = test(model=model, data=test_data, k=args.topk, device=device,
                                  train_edge_index=train_edge_index, test_edge_index=test_edge_index)
    logger.info(f'Test Interaction NDCG@{args.topk}: {test_ndcg[0]:.4f}   Test Interaction Recall@{args.topk}: {test_recall[0]:.4f}')
    logger.info(f'Test   Content   NDCG@{args.topk}: {test_ndcg[1]:.4f}   Test   Content   Recall@{args.topk}: {test_recall[1]:.4f}')
    
    logger.info('Test with Best Interaction Model')
    model = Model(data=train_data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f'{args.model_dir}interaction_{args.model_filename}'))    # Best interaction 모델 로드
    test_ndcg, test_recall = test(model=model, data=test_data, k=args.topk, device=device,
                                  train_edge_index=train_edge_index, test_edge_index=test_edge_index)
    logger.info(f'Test Interaction NDCG@{args.topk}: {test_ndcg[0]:.4f}   Test Interaction Recall@{args.topk}: {test_recall[0]:.4f}')
    logger.info(f'Test   Content   NDCG@{args.topk}: {test_ndcg[1]:.4f}   Test   Content   Recall@{args.topk}: {test_recall[1]:.4f}')
    
    logger.info('Test with Best Content Model')
    model = Model(data=train_data, emb_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f'{args.model_dir}content_{args.model_filename}'))    # Best content 모델 로드
    test_ndcg, test_recall = test(model=model, data=test_data, k=args.topk, device=device,
                                  train_edge_index=train_edge_index, test_edge_index=test_edge_index)
    logger.info(f'Test Interaction NDCG@{args.topk}: {test_ndcg[0]:.4f}   Test Interaction Recall@{args.topk}: {test_recall[0]:.4f}')
    logger.info(f'Test   Content   NDCG@{args.topk}: {test_ndcg[1]:.4f}   Test   Content   Recall@{args.topk}: {test_recall[1]:.4f}')


if __name__ == '__main__':
    main()
