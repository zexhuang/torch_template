import yaml
import logging
import torch

from pathlib import Path
from typing import Union, Optional, Callable
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torchinfo import summary
from utils.torch_tools import EarlyStopping
from utils.metric import Metrics


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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
    def fit(self, model: Union[torch.nn.Module, MessagePassing],  
            criterion: Optional[Callable]=None,  
            optimizer: Optional[torch.optim.Optimizer]=None, 
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=25,
            writer: Optional[object]=None, 
            early_stop: bool=False):
        summary(model)
        model.load_state_dict(self._load_ckpt(ckpt)['params']) if ckpt else model  
        criterion = torch.nn.CrossEntropyLoss() if criterion == None else criterion
        if not optimizer: optimizer = torch.optim.Adam(model.parameters(), 
                                                       lr=self.lr, 
                                                       weight_decay=self.w_decay) 
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                            T_0=self.epoch, 
                                                                            T_mult=1,
                                                                            eta_min=1e-6,
                                                                            verbose=True)
        writer = SummaryWriter(log_dir=f'{self.path}/runs') if writer == None else writer
        if early_stop: early_stopping = EarlyStopping(patience=self.patience)
         
        for ep in range(1, self.epoch+1):
            t_ls = self._fit_impl(model, optimizer, criterion, train_loader)
            writer.add_scalar('Loss/train', t_ls, ep)
            v_ls = self._val_impl(model, criterion, val_loader) 
            writer.add_scalar('Loss/val', v_ls, ep)
            # Adjust learning rate 
            lr_scheduler.step()
            writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            
            if early_stop:
                early_stopping(v_ls.item()) # accpet float type
                if early_stopping.counter == 0: 
                    self._save_ckpt(model, ckpt_name=f'best_at_epoch{ep}')
                if early_stopping.stop: break
                
            if ep % save_period == 0: # save model at every n epoch
                self._save_ckpt(model, ckpt_name=f'epoch{ep}')
            
    def _fit_impl(self, model, optimizer, criterion, dataloader):
        model.train()
        model.to(self.device)
        ls = 0.0
        for data in dataloader:
            optimizer.zero_grad()         # Clear gradients
            data.to(torch.device(self.device))
            logits = model(data) 
            y = data['y'].to(torch.device(self.device), dtype=torch.long)
            loss = criterion(logits, y)   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            # Loss dim reduction="mean"
            ls += len(data) * loss.detach().clone()  
        return ls / len(dataloader.dataset)
    
    def _val_impl(self, model, criterion, dataloader):
        model.eval()
        model.to(self.device)
        ls = 0.0
        for data in dataloader:
            data.to(torch.device(self.device))
            logits = model(data) 
            y = data['y'].to(torch.device(self.device), dtype=torch.long)
            loss = criterion(logits, y)   
            ls += len(data) * loss.detach().clone()
        return ls / len(dataloader.dataset)

    def predict(self, model: Union[torch.nn.Module, MessagePassing], 
                dataloader: Optional[DataLoader]=None, 
                metric: Optional[Callable]=None,
                ckpt: Union[str, Path, None]=None):
        model.load_state_dict(self._load_ckpt(ckpt)['params']) if ckpt else model  
        metric = Metrics(self.num_cls) if metric == None else metric
        return self._pred_impl(model, metric, dataloader), model
    
    def _pred_impl(self, model, metric, dataloader):
        model.eval()
        model.to(self.device)
        for data in dataloader:
            data.to(torch.device(self.device))
            logits = model(data) 
            y = data['y'].to(torch.device(self.device), dtype=torch.long)
            metric.update(logits, y)
        return metric  
    
    def _save_ckpt(self, model, ckpt_name):
        path = Path(self.path) / 'ckpt'
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({'params':model.state_dict()}, path.joinpath(ckpt_name))
        logging.info('model ckpt is saved.')
    
    def _load_ckpt(self, ckpt_name):
        path = Path(self.path) / 'ckpt'
        return torch.load(path.joinpath(ckpt_name)) # {'params': Tensor}