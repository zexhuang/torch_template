import torch
import numpy as np


class Metrics(object):
    def __init__(self, num_cls):
        self.num_cls = num_cls
        self.cm = torch.zeros(self.num_cls,
                              self.num_cls,
                              dtype=torch.long)

    def update(self, pred, target):
        target = target.detach().cpu().numpy()
        pred = torch.argmax(pred, 1)
        pred = pred.detach().cpu().numpy()
        # Bin-count trick
        label = self.num_cls * target.astype('int') + pred.astype('int')
        count = np.bincount(label, minlength=self.num_cls ** 2)
        cm = count.reshape(self.num_cls, self.num_cls)
        self.cm += cm

    def acc(self):
        return (self._tp_tn() / self._sum()).item() * 100
        
    def miou(self):
        iou = self.iou()
        return (iou.sum() / len(iou)).item() * 100
    
    def iou(self):
        return torch.diagonal(self.cm, 0) / (self.cm.sum(dim=0) 
                                             + self.cm.sum(dim=1) 
                                             - torch.diagonal(self.cm, 0))
    
    def _tp_tn(self):
        return torch.sum(torch.diagonal(self.cm, 0))
    
    def _fn(self):
        return torch.sum(torch.triu(self.cm, 1))
    
    def _fp(self):
        return torch.sum(torch.triu(self.cm.t(), 1))
    
    def _sum(self):
        return torch.sum(self.cm)