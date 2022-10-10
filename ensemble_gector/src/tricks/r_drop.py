import torch.nn.functional as F




def compute_kl_loss(self, p, q, pad_mask=None):
    """useage
    # define your task model, which outputs the classifier logits
    model = TaskModel()
    # keep dropout and forward twice
    logits = model(x)

    logits2 = model(x)

    # cross entropy loss for classifier
    ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

    kl_loss = compute_kl_loss(logits, logits2)

    # carefully choose hyper-parameters
    loss = ce_loss + α * kl_loss
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

