import yaml
import logging
import torch

from pathlib import Path
from typing import Union, Optional, Callable, Dict
from tqdm import tqdm

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import Precision, Recall, F1Score, JaccardIndex

from utils.utils import EarlyStopping


class Trainer:
    def __init__(self, cfg:Union[str, Path, Dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, Dict):
            self.cfg = cfg
        else:
            raise ValueError("trainer cfg must be a string, Path, or dictionary.")
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        
        self.metrics = {
            'Precision': Precision(task='binary', threshold=0.5).to(self.cfg['device']),
            'Recall': Recall(task='binary', threshold=0.5).to(self.cfg['device']),
            'F1Score': F1Score(task='binary', threshold=0.5).to(self.cfg['device']),
            'mIoU': JaccardIndex(task='binary', threshold=0.5).to(self.cfg['device'])
        }
        
        self.set_seed()

    def fit(self, model: torch.nn.Module, 
            criterion: Optional[Callable]=None,  
            optimizer: Optional[torch.optim.Optimizer]=None, 
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=1):
        summary(model)
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False)
        
        criterion = criterion or torch.nn.CrossEntropyLoss()

        optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.w_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.epoch,
            T_mult=1,
            eta_min=1e-6,
            verbose=False
        )
        
        writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)
         
        for ep in tqdm(range(1, self.epoch + 1)):
            t_loss, t_mcs = self._fit_impl(model, optimizer, criterion, train_loader)
            writer.add_scalar('Loss/train', t_loss, ep)
            print('Loss/train', f"{t_loss:.2f}", end=' -- ')
            
            for mc_name, mc_value in t_mcs.items():
                writer.add_scalar(f'{mc_name}/train', mc_value, ep)
                print(mc_name+'/train', f"{mc_value * 100:.2f}", end=' -- ')

            # Adjust learning rate 
            lr_scheduler.step()
            writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
    
            if val_loader:
                v_loss, v_mcs = self._val_impl(model, criterion, val_loader) 
                writer.add_scalar('Loss/val', v_loss, ep)
                print('Loss/val', f"{v_loss:.2f}", end=' -- ')
                
                for mc_name, mc_value in v_mcs.items():
                    writer.add_scalar(f'{mc_name}/val', mc_value, ep)
                    print(mc_name+'/val', f"{mc_value * 100:.2f}", end=' -- ')
                    
                if ep % save_period == 0: # save model at every n epoch  
                    early_stopping(v_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    
                    if early_stopping.early_stop: 
                        break

    def _fit_impl(self, model, optimizer, criterion, dataloader):
        model.train()
        ls = 0.0
        self._reset_metrics()
        
        for step, data in enumerate(dataloader):
            label, input = data['label'].to(self.cfg['device']), data['input'].to(self.cfg['device'])
            optimizer.zero_grad()         # Clear gradients
            out = model(input)
            loss = criterion(out, label)  # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            
            ls += loss.item() * out.size(0)
            self._update_metrics(out, label)
            
        per_sample_loss = ls / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return per_sample_loss, mcs
    
    def _val_impl(self, model, criterion, dataloader):
        model.eval()
        ls = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                label, input = data['label'].to(self.cfg['device']), data['input'].to(self.cfg['device'])
                out = model(input)
                loss = criterion(out, label)   
                ls += loss.item() * out.size(0)
                self._update_metrics(out, label)
            
        per_sample_loss = ls / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return per_sample_loss, mcs

    def eval(self, model: Union[torch.nn.Module], 
             criterion: Optional[Callable]=None,  
             dataloader: Optional[DataLoader]=None, 
             ckpt: Union[str, Path, None]=None,
             verbose: bool=False):
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'])
            
        criterion = criterion or torch.nn.CrossEntropyLoss()

        e_loss, e_mcs = self._eval_impl(model, criterion, dataloader) 
        
        if verbose:
            summary(model)
            
            print('Loss/eval', f"{e_loss:.2f}", end=' -- ')
            for mc_name, mc_value in e_mcs.items():
                print(mc_name+'/eval', f"{mc_value * 100:.2f}", end=' -- ')
                
        return e_mcs

    def _eval_impl(self, model, criterion, dataloader):
        model.eval()
        ls = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                label, input = data['label'].to(self.cfg['device']), data['input'].to(self.cfg['device'])
                out = model(input)
                loss = criterion(out, label)   
                ls += loss.item() * out.size(0)
                self._update_metrics(out, label)
            
        per_sample_loss = ls / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return per_sample_loss, mcs
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False) 
        model.eval()
        model.to(self.cfg['device'])
        return model
    
    def set_seed(self, seed = 3407):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility
        
    def _save_ckpt(self, model, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt'
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path / ckpt_name)
        logging.info(f'Model checkpoint saved at {ckpt_path / ckpt_name}')
        
    def _load_ckpt(self, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt' / ckpt_name 
        return torch.load(ckpt_path)
    
    def _update_metrics(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target.long())
            
    def _compute_metrics(self):
        return [metric.compute().item() for metric in self.metrics.values()]
        
    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()