import torch
import torch.nn.functional as F


def sent_token_simloss(y_pred, 
                       sim_labels,
                       input_ids, 
                       gamma=2, 
                       pos_id=0, 
                       neg_id=1,
                       pos_weight=1,
                       neg_weight=2,
                       sos_token_id=101,
                       eos_token_id=102,
                       ignore_idx=-100):
    
    """句子向量(cls)与字向量的相似度,正确的字相似,不正确的字不相似

    Args:
        y_pred (_type_): with cls, sep,  (b, l, h) 
        sim_labels (_type_): b, l: 0  match pos_id, neg_id
        input_ids: for find eos_token_id,
        gamma (int, optional): _description_. 超参数 to 20.
        pos_id (int, optional): _description_. 正例的label to 0.
        neg_id (int, optional): _description_. 负例的label to 1.
        pow_weight (int, optional): _description_. Defaults to 1.
        neg_weight (int, optional): _description_. Defaults to 3.
        ignore_idx (int, optional): _description_. Defaults to -100.
    """
    device = sim_labels.device
    cls2token_cos_similarity = F.cosine_similarity(y_pred[input_ids==sos_token_id].unsqueeze(1), y_pred, dim=2) 
    sep2token_cos_similarity = F.cosine_similarity(y_pred[input_ids==eos_token_id].unsqueeze(1), y_pred, dim=2) 

    # compute weight params, keep ignore index
    sim_labels = torch.where(sim_labels == pos_id, -pos_weight, sim_labels)
    sim_labels = torch.where(sim_labels == neg_id, neg_weight, sim_labels)
    
    
    cls2token_cos_similarity = gamma* cls2token_cos_similarity * sim_labels
    sep2token_cos_similarity = gamma* sep2token_cos_similarity * sim_labels
    
    # permute
    cls2token_cos_similarity, sep2token_cos_similarity, sim_labels = \
    cls2token_cos_similarity.view(-1), sep2token_cos_similarity.view(-1), sim_labels.view(-1)
    
    # ignore index
    cls2token_cos_similarity = cls2token_cos_similarity[sim_labels!=ignore_idx]
    sep2token_cos_similarity = sep2token_cos_similarity[sim_labels!=ignore_idx]
    
    # log(sum(1+exp(sim)))
    cls2token_cos_similarity = torch.cat((torch.tensor([0]).float().to(device), cls2token_cos_similarity), dim=0).contiguous()  # 这里加0是因为e^0 = 1相当于在logsumexp中加了1
    sep2token_cos_similarity = torch.cat((torch.tensor([0]).float().to(device), sep2token_cos_similarity), dim=0).contiguous()  # 这里加0是因为e^0 = 1相当于在logsumexp中加了1
    
    loss = torch.logsumexp(cls2token_cos_similarity, dim=0) + torch.logsumexp(sep2token_cos_similarity, dim=0)
    
    return loss
    

if __name__ == '__main__':
    y = torch.randn(2, 4, 768)
    slabels = torch.LongTensor(
        [
            [-100,0,1,-100],
            [-100,0,-100,-100],
        ]
    )
    
    input_ids = torch.LongTensor(
        [
            [101,5,6,102],
            [101,1, 102,-100],
            
        ]
    )
    sent_token_simloss(y, slabels, input_ids)
