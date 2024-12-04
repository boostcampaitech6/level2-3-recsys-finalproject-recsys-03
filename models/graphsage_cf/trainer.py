import torch
from tqdm import tqdm
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn import MIPSKNNIndex
from torch_geometric.metrics import LinkPredNDCG, LinkPredRecall


def train(args, model, optimizer, loss, dataloader, data, train_edge_index, k, device):
    model.train()

    data = data.to(device)
    
    # negative sampling을 위해 노드 개수 저장
    num_track = data['track'].num_nodes
    
    # loss 합산을 위한 변수
    total_loss = 0
    num_batches = 0

    # train 진행
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # batch 데이터 임베딩
        embeddings = model(data)
        
        # negative sampling 후 loss 계산
        batch_loss = 0
        
        # (user, listen, track) edge loss
        pos_edge_index = data['user', 'listen', 'track'].edge_index[:, batch['user', 'listen', 'track'].e_id].to(device)
        for _ in range(args.negative_sampling):
            _, _, neg_tail_tensor = structured_negative_sampling(edge_index=pos_edge_index,
                                                                 num_nodes=num_track,
                                                                 contains_neg_self_loops=False)
            anchor_emb = embeddings['user'][pos_edge_index[0]]
            positive_emb = embeddings['track'][pos_edge_index[1]]
            negative_emb = embeddings['track'][neg_tail_tensor]
            loss_tmp = loss(anchor_emb, positive_emb, negative_emb)
            batch_loss += loss_tmp
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
    average_loss = total_loss / num_batches
    
    # train 평가
    ndcg_scores_dict, recall_scores_dict = test(model=model, data=data, k=k, device=device, train_edge_index=train_edge_index, test_edge_index=train_edge_index, train_mode=True)
    
    return ndcg_scores_dict, recall_scores_dict, average_loss


@torch.no_grad()
def test(model, data, k, device, train_edge_index, test_edge_index, train_mode=False):
    model.eval()

    embeddings = model(data.to(device))
    
    # 학습된 노드 임베딩
    user_emb = embeddings['user']
    track_emb = embeddings['track']
    

    # Content 성능 평가
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    
    # MIPS(maximum inner product search) 기반 KNN
    if train_mode:    # train 상황에는 train 데이터만 입력하여 예측
        _, pred_index_mat = MIPSKNNIndex(track_emb).search(user_emb, k)    # user 정보를 기반으로 track 추천
    else:    # test 상황에는 train 데이터를 제외하고 예측
        _, pred_index_mat = MIPSKNNIndex(track_emb).search(user_emb, k, exclude_links=train_edge_index)
    
    ndcg_metric = LinkPredNDCG(k=k).to(device)
    recall_metric = LinkPredRecall(k=k).to(device)
    
    ndcg_metric.update(pred_index_mat, test_edge_index)
    recall_metric.update(pred_index_mat, test_edge_index)
    
    ndcg_score = float(ndcg_metric.compute())
    recall_score = float(recall_metric.compute())

    return ndcg_score, recall_score


def feature(data, k, device, train_edge_index, valid_edge_index, test_edge_index):
    ndcg_scores_dict = dict()
    recall_scores_dict = dict()
    
    # 학습 전 특성 정보
    user_feat = data['user'].x
    track_feat = data['track'].x
    
    user_feat = user_feat.to(device)
    track_feat = track_feat.to(device)
    train_edge_index = train_edge_index.to(device)
    valid_edge_index = valid_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    
    # Feature 성능 평가
    for mode, edge_index in zip(['train','valid','test'], [train_edge_index, valid_edge_index, test_edge_index]):
        if mode == 'train':
            _, pred_index_mat = MIPSKNNIndex(track_feat).search(user_feat, k)
        else:
            _, pred_index_mat = MIPSKNNIndex(track_feat).search(user_feat, k, exclude_links=train_edge_index)
        
        ndcg_metric = LinkPredNDCG(k=k).to(device)
        recall_metric = LinkPredRecall(k=k).to(device)
        
        ndcg_metric.update(pred_index_mat, edge_index)
        recall_metric.update(pred_index_mat, edge_index)
        
        ndcg_score = float(ndcg_metric.compute())
        recall_score = float(recall_metric.compute())
        
        ndcg_scores_dict[mode] = ndcg_score
        recall_scores_dict[mode] = recall_score
    
    return ndcg_scores_dict, recall_scores_dict
