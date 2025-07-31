import yaml
import logging
import random
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchinfo import summary
from tqdm import tqdm

from utils.utils import EarlyStopping


class BaseTrainer:
    def __init__(self, cfg: Union[str, Path, dict]):
        if isinstance(cfg, (str, Path)):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)
        elif isinstance(cfg, dict):
            self.cfg = cfg
        else:
            raise ValueError("Config must be a string, Path, or dictionary.")

        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.num_cls = self.cfg['out_channels']
        self.device = self.cfg['device']

        self.set_seed()

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        ckpt = Path(ckpt) if isinstance(ckpt, str) else ckpt
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])
        model.eval().to(self.device)
        return model

    def _save_ckpt(self, model, ckpt_name):
        ckpt_dir = Path(self.path) / 'ckpt'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'params': model.state_dict()}, ckpt_dir / ckpt_name)
        logging.info(f'Model checkpoint saved: {ckpt_name}')

    def _load_ckpt(self, ckpt_name, device):
        ckpt_path = Path(self.path) / 'ckpt' / ckpt_name
        return torch.load(ckpt_path, map_location=device, weights_only=True)


class Trainer(BaseTrainer):
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],
            criterion: Optional[Callable] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            ckpt: Union[str, Path, None] = None,
            save_period: int = 10):

        summary(model, depth=3)
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        criterion = criterion or torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch, eta_min=1e-5)

        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)

        for ep in tqdm(range(1, self.epoch + 1)):
            train_loss = self._train_one_epoch(model, optimizer, criterion, train_loader)
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            lr_scheduler.step()

            if ep % save_period == 0:
                if val_loader:
                    val_loss = self._evaluate(model, criterion, val_loader)
                    self.writer.add_scalar('Loss/val', val_loss, ep)
                    early_stopping(val_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping at epoch {ep}.")
                        break
                else:
                    self._save_ckpt(model, ckpt_name=f'epoch{ep}')

    def _train_one_epoch(self, model, optimizer, criterion, dataloader):
        model.train().to(self.device)
        total_loss = 0.0

        for data in dataloader:
            data = data.to(self.device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out['y'], data['y'])
            loss.backward()
            optimizer.step()
            total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def _evaluate(self, model, criterion, dataloader):
        model.eval().to('cpu')
        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                data = data.to('cpu')
                out = model(data)
                loss = criterion(out['y'], data['y'])
                total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing],
             dataloader: DataLoader,
             metric: Optional[Dict[str, Metric]] = None,
             ckpt: Union[str, Path, None] = None,
             verbose: bool = False):

        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        metric = metric or {
            'OA': BinaryAccuracy(),
            'mIoU': BinaryJaccardIndex()
        }

        model.eval().to(self.device)
        for cm in metric.values():
            cm.to(self.device)

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                out = model(data)
                for name, cm in metric.items():
                    cm.update(out['y'], data['y'].long())

        if verbose:
            for name, cm in metric.items():
                print(f"{name}: {cm.compute().cpu().item()}")

        return metric
