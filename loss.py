import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, t):
        t = t.squeeze(1)
        t = t.long()
        return self.ce_loss(pred, t)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, pred, t):
        t = t.squeeze(1)
        t = t.long()
        return self.ce_loss(pred, t)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, true, eps=1e-7):
        device = logits.device  # Get the device of the logits tensor
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1, device=device)[
                true.squeeze(1).long()
            ]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = torch.nn.functional.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
        return 1 - dice_loss
