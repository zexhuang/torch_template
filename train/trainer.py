import yaml
import torch

from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, Callable, Dict

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from torch.utils.tensorboard import SummaryWriter
from utils.torch_tools import EarlyStopping
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchinfo import summary


class Trainer:
    def __init__(self, cfg:Union[str, Path, dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, dict):
            self.cfg = cfg
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.num_cls = self.cfg['num_cls']
        self.device = self.cfg['device']
            
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],  
            criterion: Optional[Callable]=None,  
            optimizer: Optional[torch.optim.Optimizer]=None, 
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=10,
            writer: Optional[object]=None, 
            early_stop: bool=False):
        summary(model, depth=100)
        
        model.to(self.device)  
        model.load_state_dict(self._load_ckpt(ckpt)['params']) if ckpt else model  
        
        criterion = torch.nn.CrossEntropyLoss() if criterion == None else criterion
        
        if not optimizer: optimizer = torch.optim.Adam(model.parameters(), 
                                                       lr=self.lr, 
                                                       weight_decay=self.w_decay) 
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                            T_0=self.epoch, 
                                                                            T_mult=1,
                                                                            eta_min=1e-6)
        
        writer = SummaryWriter(log_dir=f'{self.path}/runs') if writer == None else writer
        
        if early_stop: early_stopping = EarlyStopping(path=self.path, patience=self.patience)
         
        for ep in tqdm(range(1, self.epoch+1)):
            t_ls = self._fit_impl(model, optimizer, criterion, train_loader)
            writer.add_scalar('Loss/train', t_ls, ep)
            
            # Adjust learning rate 
            lr_scheduler.step()
            writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            
            if ep % save_period == 0: # save model at every n epoch
                if val_loader:
                    v_ls = self._val_impl(model, criterion, val_loader) 
                    writer.add_scalar('Loss/val', v_ls, ep)
                    
                    if early_stop:
                        early_stopping(v_ls.item(), model, optimizer, ep, lr_scheduler.get_last_lr()) # accpet float type
                        if early_stopping.early_stop: break
            
    def _fit_impl(self, model, optimizer, criterion, dataloader):
        model.train()
        ls = 0.0
        for _, data in enumerate(dataloader):
            optimizer.zero_grad()         # Clear gradients
            data.to(self.device)
            logits = model(data) 
            loss = criterion(logits, data['y'])   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            # Loss dim reduction="mean"
            ls += len(data) * loss.detach().clone()  
        return ls / len(dataloader.dataset)
    
    def _val_impl(self, model, criterion, dataloader):
        model.eval()
        ls = 0.0
        for _, data in enumerate(dataloader):
            data.to(self.device)
            logits = model(data) 
            loss = criterion(logits, data['y'])   
            ls += len(data) * loss.detach().clone()
        return ls / len(dataloader.dataset)
    
    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing], 
             dataloader: Optional[DataLoader]=None, 
             metric: Optional[Dict[str, Metric]]=None,
             ckpt: Union[str, Path, None]=None,
             verbose: bool=False):
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['model_state_dict']) if ckpt else model  
        model.to(self.device)
        
        if metric is None:
            metric = {'OA': BinaryAccuracy(), 
                      'mIoU': BinaryJaccardIndex()}
            
        for name, cm in metric.items(): 
            cm.to(self.device)

        metric = self._eval_impl(model, metric, dataloader)
        
        if verbose:
            for name, cm in metric.items(): print(f"{name}: {cm.compute().cpu().numpy().tolist()}")    
        return metric
    
    def _eval_impl(self, model, metric, dataloader):
        model.eval()
        for data in dataloader:
            data.to(self.device)
            logits = model(data) 
            for name, cm in metric.items():
                cm.update(logits, data['y'])
        return metric 
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['model_state_dict']) 
        model.eval()
        model.to(self.device)
        return model
    
    def _load_ckpt(self, ckpt_name, device):
        path = Path(self.path) / 'ckpt'
        return torch.load(path.joinpath(ckpt_name), map_location=device)