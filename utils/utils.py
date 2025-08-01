import torch
from pathlib import Path


# Code adapoted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation monitor doesn't improve after a given patience."""
    def __init__(self, path, best_score=None, patience=10, delta=0.0, verbose=False, trace_func=print):
        """
        Args:
            path (str): Path for the checkpoint to be saved to.
            best_score (float or none): Value of metric of the best model.
                            Default: None
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            trace_func (function): trace print function.
                            Default: print            
        """
        self.path = Path(path)
        self.best_score = best_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model, optimizer, epoch, last_lr, cm=None):
        score = loss 
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'params': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': last_lr[0],
            'cm': cm
        }
    
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                self.trace_func(
                    f"\n[EarlyStopping] Initializing best score.\n"
                    f"    Starting Score: {score:.6f}\n"
                    f"    Saving initial checkpoint to: {self.path / 'ckpt' / 'best_val_epoch.pth'}\n"
                )
            self.save_checkpoint(checkpoint, score)
            
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"\n[EarlyStopping] Validation loss did not improve.\n"
                    f"    Best Score     : {self.best_score:.6f}\n"
                    f"    Current Score  : {score:.6f}\n"
                    f"    Delta Threshold: {self.delta}\n"
                    f"    Counter        : {self.counter} / {self.patience}\n"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                self.trace_func(
                    f"\n[EarlyStopping] Validation loss improved!\n"
                    f"    Previous Best : {self.best_score:.6f}\n"
                    f"    New Best      : {loss:.6f}\n"
                    f"    Saving checkpoint to: {self.path / 'ckpt' / 'best_val_epoch.pth'}\n"
                )
            self.save_checkpoint(checkpoint, score)
            self.counter = 0

    def save_checkpoint(self, checkpoint, loss):
        '''Saves model when validation loss decrease.'''
        checkpoint_path = self.path / 'ckpt'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path.joinpath('best_val_epoch.pth'))
        self.best_score = loss
