import torch.nn.functional as F
import torch
from .base import BaseLoss

class_labels = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]



class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
    
    
class DirectionCrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(DirectionCrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        # print(labels.shape)
        n, c, p = logits.size()
        logits = logits.float()
        # labels = labels.unsqueeze(1)
        labels = torch.tensor([class_labels.index(label) for label in labels]).unsqueeze(1) # Convert labels to indices
        labels = labels.to(logits.device)
        
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        self.info.update({'direction_loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'direction_accuracy': accu})
        return loss, self.info