import torch
from tqdm import tqdm
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn import MIPSKNNIndex
from torch_geometric.metrics import LinkPredNDCG, LinkPredRecall


def train(args, model, optimizer, loss, dataloader, data, train_interaction_edge_index, train_content_edge_index, k, device):
    model.train()

    data = data.to(device)
    
    # negative sampling을 위해 노드 개수 저장
    num_track = data['track'].num_nodes
    num_artist = data['artist'].num_nodes
    num_tag = data['tag'].num_nodes
    
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
        
        # (user, listen, track) edge
        pos_edge_listen_index = data['user', 'listen', 'track'].edge_index[:, batch['user', 'listen', 'track'].e_id].to(device)
        neg_edge_listen_index = torch.empty(0, dtype=torch.int64).to(device)    # 원소가 정수형인 빈 tensor
        for _ in range(args.negative_sampling):
            head_tensor, _, neg_tail_tensor = structured_negative_sampling(edge_index=pos_edge_listen_index,
                                                                           num_nodes=num_track,
                                                                           contains_neg_self_loops=False)    # output : (head_tensor, pos_tail_tensor, neg_tail_tensor)
            neg_edge_listen_index = torch.cat((neg_edge_listen_index, torch.vstack((head_tensor, neg_tail_tensor))), dim=1)    # negative sampling 개수만큼 누적
        edge_listen_index = torch.cat((pos_edge_listen_index, neg_edge_listen_index), dim=1)    # positive edge와 negative edge 연결
        edge_listen_user_emb = embeddings['user'][edge_listen_index[0]].to(device)    # edge에 대한 user 임베딩
        edge_listen_track_emb = embeddings['track'][edge_listen_index[1]].to(device)    # edge에 대한 track 임베딩
        batch_loss += loss(edge_listen_user_emb, edge_listen_track_emb)
        
        # # (track, sungby, artist) edge
        # pos_edge_sungby_index = data['track', 'sungby', 'artist'].edge_index[:, batch['track', 'sungby', 'artist'].e_id].to(device)
        # neg_edge_sungby_index = torch.empty(0, dtype=torch.int64).to(device)
        # for _ in range(args.negative_sampling):
        #     head_tensor, _, neg_tail_tensor = structured_negative_sampling(edge_index=pos_edge_sungby_index,
        #                                                                    num_nodes=num_artist,
        #                                                                    contains_neg_self_loops=False)
        #     neg_edge_sungby_index = torch.cat((neg_edge_sungby_index, torch.vstack((head_tensor, neg_tail_tensor))), dim=1)
        # edge_sungby_index = torch.cat((pos_edge_sungby_index, neg_edge_sungby_index), dim=1)
        # edge_sungby_track_emb = embeddings['track'][edge_sungby_index[0]].to(device)
        # edge_sungby_artist_emb = embeddings['artist'][edge_sungby_index[1]].to(device)
        # batch_loss += loss(edge_sungby_track_emb, edge_sungby_artist_emb)
        
        # (track, tagged, tag) edge
        pos_edge_tagged_index = data['track', 'tagged', 'tag'].edge_index[:, batch['track', 'tagged', 'tag'].e_id].to(device)
        neg_edge_tagged_index = torch.empty(0, dtype=torch.int64).to(device)
        for _ in range(args.negative_sampling):
            head_tensor, _, neg_tail_tensor = structured_negative_sampling(edge_index=pos_edge_tagged_index,
                                                                           num_nodes=num_tag,
                                                                           contains_neg_self_loops=False)
            neg_edge_tagged_index = torch.cat((neg_edge_tagged_index, torch.vstack((head_tensor, neg_tail_tensor))), dim=1)
        edge_tagged_index = torch.cat((pos_edge_tagged_index, neg_edge_tagged_index), dim=1)
        edge_tagged_track_emb = embeddings['track'][edge_tagged_index[0]].to(device)
        edge_tagged_tag_emb = embeddings['tag'][edge_tagged_index[1]].to(device)
        batch_loss += loss(edge_tagged_track_emb, edge_tagged_tag_emb)
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
    average_loss = total_loss / num_batches
    
    # train 평가
    ndcg_scores, recall_scores = test(model=model, data=data, k=k, device=device, train_mode=True,
                                      train_interaction_edge_index=train_interaction_edge_index, train_content_edge_index=train_content_edge_index)
    
    return ndcg_scores, recall_scores, average_loss


@torch.no_grad()
def test(model, data, k, device, train_mode=False, interaction_mode=True, content_mode=True,
         train_interaction_edge_index=None, test_interaction_edge_index=None, train_content_edge_index=None, test_content_edge_index=None):
    model.eval()

    embeddings = model(data.to(device))
    
    # 학습된 노드 임베딩
    user_emb = embeddings['user']
    track_emb = embeddings['track']
    tag_emb = embeddings['tag']
    
    ndcg_scores = [0, 0]
    recall_scores = [0, 0]
    
    # Interaction 성능 평가
    if interaction_mode:
        train_interaction_edge_index = train_interaction_edge_index.to(device)
        if not train_mode:    # train 상황에는 제외
            test_interaction_edge_index = test_interaction_edge_index.to(device)
        
        # MIPS(maximum inner product search) 기반 KNN
        if train_mode:    # train 상황에는 train 데이터만 입력하여 예측
            _, pred_interaction_index_mat = MIPSKNNIndex(track_emb).search(user_emb, k)    # user 정보를 기반으로 track 추천
        else:    # test 상황에는 train 데이터를 제외하고 예측
            _, pred_interaction_index_mat = MIPSKNNIndex(track_emb).search(user_emb, k, exclude_links=train_interaction_edge_index)
        
        # 평가 지표
        ndcg_interaction_metric = LinkPredNDCG(k=k).to(device)
        recall_interaction_metric = LinkPredRecall(k=k).to(device)
        
        # train에서 평가하는 데이터와 validation/test에서 평가하는 데이터 구분
        if train_mode:
            ndcg_interaction_metric.update(pred_interaction_index_mat, train_interaction_edge_index)
            recall_interaction_metric.update(pred_interaction_index_mat, train_interaction_edge_index)
        else:
            ndcg_interaction_metric.update(pred_interaction_index_mat, test_interaction_edge_index)
            recall_interaction_metric.update(pred_interaction_index_mat, test_interaction_edge_index)
        
        ndcg_interaction_score = float(ndcg_interaction_metric.compute())
        recall_interaction_score = float(recall_interaction_metric.compute())
        
        ndcg_scores[0] = ndcg_interaction_score
        recall_scores[0] = recall_interaction_score

    # Content 성능 평가
    if content_mode:
        train_content_edge_index = train_content_edge_index.to(device)
        if not train_mode:
            test_content_edge_index = test_content_edge_index.to(device)
        
        if train_mode:
            _, pred_content_index_mat = MIPSKNNIndex(track_emb).search(tag_emb, k)    # tag 정보를 기반으로 track 추천
        else:
            _, pred_content_index_mat = MIPSKNNIndex(track_emb).search(tag_emb, k, exclude_links=train_content_edge_index)
        
        ndcg_content_metric = LinkPredNDCG(k=k).to(device)
        recall_content_metric = LinkPredRecall(k=k).to(device)
        
        if train_mode:
            ndcg_content_metric.update(pred_content_index_mat, train_content_edge_index)
            recall_content_metric.update(pred_content_index_mat, train_content_edge_index)
        else:
            ndcg_content_metric.update(pred_content_index_mat, test_content_edge_index)
            recall_content_metric.update(pred_content_index_mat, test_content_edge_index)
        
        ndcg_content_score = float(ndcg_content_metric.compute())
        recall_content_score = float(recall_content_metric.compute())

        ndcg_scores[1] = ndcg_content_score
        recall_scores[1] = recall_content_score
    
    return ndcg_scores, recall_scores
