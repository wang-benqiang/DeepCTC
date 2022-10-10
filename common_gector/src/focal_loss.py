import torch


class FocalCELoss:
    def __init__(self, loss_labels_weights=None, gamma=2, ignore_index=-100):
        """[summary]

        Args:
            loss_labels_weights ([type], optional): [对应类别的loss权重比例]. Defaults to None.
            gamma (int, optional): [description]. Defaults to 2.
            ignore_index (int, optional): [忽略idx的]. Defaults to -1.
        """
        self.loss_labels_weights = torch.FloatTensor(loss_labels_weights) if loss_labels_weights else None
        self.gamma = gamma
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        if self.loss_labels_weights is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(
                outputs, targets, ignore_index=self.ignore_index)

        elif self.loss_labels_weights is not None and self.gamma == 0:
            if self.loss_labels_weights.device != outputs.device:
                self.loss_labels_weights = self.loss_labels_weights.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                weight=self.loss_labels_weights,
                ignore_index=self.ignore_index)

        elif self.loss_labels_weights is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                reduction='none',
                ignore_index=self.ignore_index)
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t)**self.gamma * ce_loss).mean()

        elif self.loss_labels_weights is not None and self.gamma != 0:
            if self.loss_labels_weights.device != outputs.device:
                self.loss_labels_weights = self.loss_labels_weights.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                reduction='none',
                ignore_index=self.ignore_index)
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                weight=self.loss_labels_weights,
                reduction='none',
                ignore_index=self.ignore_index)
            focal_loss = ((1 - p_t)**self.gamma *
                          ce_loss).mean()  # mean over the batch

        return focal_loss


# if __name__ == '__main__':
#     outputs = torch.tensor([[2, 1., 3], [2.5, 1, 3]], device='cpu')
#     targets = torch.tensor([0, 2], device='cpu')
#     print(torch.nn.functional.softmax(outputs, dim=1))

#     fl = FocalLoss([0.5, 1, 1], 2)

#     print(fl(outputs, targets))
