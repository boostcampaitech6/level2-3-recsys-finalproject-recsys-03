import torch
from tqdm import tqdm
from torch_geometric.nn import MIPSKNNIndex
from torch_geometric.metrics import LinkPredNDCG, LinkPredRecall


def train(model, optimizer, loss, dataloader, data, train_edge_index_eval, k, device):
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    # train 진행
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        embeddings = model(batch)
        
        edge_user_emb = embeddings['user'][batch['user', 'listen', 'track'].edge_label_index].to(device)
        edge_track_emb = embeddings['track'][batch['user', 'listen', 'track'].edge_label_index].to(device)

        batch_loss = loss(edge_user_emb, edge_track_emb)
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    average_loss = total_loss / num_batches
    
    # train 평가
    ndcg_score, recall_score = test(model=model, data=data, k=k, device=device,
                                    train_mode=True, train_edge_index_eval=train_edge_index_eval)
    
    return ndcg_score, recall_score, average_loss


@torch.no_grad()
def test(model, data, k, device, train_mode=False, train_edge_index_eval=None, test_edge_index_eval=None):
    model.eval()

    embeddings = model(data.to(device))
    
    user_emb = embeddings['user']
    track_emb = embeddings['track']
    
    train_edge_index_eval = train_edge_index_eval.to(device)
    if not train_mode:
        test_edge_index_eval = test_edge_index_eval.to(device)
    
    # MIPS(maximum inner product search) 기반 KNN
    mips = MIPSKNNIndex(track_emb)
    if train_mode:
        _, pred_index_mat = mips.search(user_emb, k)
    else:
        _, pred_index_mat = mips.search(user_emb, k, exclude_links=train_edge_index_eval)
    
    # 평가 지표
    ndcg_metric = LinkPredNDCG(k=k).to(device)
    recall_metric = LinkPredRecall(k=k).to(device)
    
    # train에서 평가하는 데이터와 validation/test에서 평가하는 데이터 구분
    if train_mode:
        ndcg_metric.update(pred_index_mat, train_edge_index_eval)
        recall_metric.update(pred_index_mat, train_edge_index_eval)
    else:
        ndcg_metric.update(pred_index_mat, test_edge_index_eval)
        recall_metric.update(pred_index_mat, test_edge_index_eval)
    
    ndcg_score = float(ndcg_metric.compute())
    recall_score = float(recall_metric.compute())
    
    return ndcg_score, recall_score
