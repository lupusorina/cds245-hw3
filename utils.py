import torch
import numpy as np

class EarlyStopping:
    def __init__(self,
                 patience: int = 7,
                 verbose: bool = False,
                 delta: float = 0,
                 ):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.counter = 0

    def __call__(self,
                 val_loss: float,
                 ):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

