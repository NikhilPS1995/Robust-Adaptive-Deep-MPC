import numpy as np
import torch
from utils.utils import torchify

class SimDataset:
    def __init__(self, xt, wt, ut, fU_t, fL_t, gU_t, gL_t, fBU_t, fBL_t, x_tp1, w_tp1):
        self.input = np.hstack([xt, wt, ut, fU_t, fL_t, gU_t, gL_t, fBU_t, fBL_t])
        self.target = w_tp1 #np.hstack([x_tp1, w_tp1])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __getitem__(self, idx):
        
        x = self.input[idx, :]
        target = self.target[idx, :]
        return torchify(x, target, device=self.device)

    def __len__(self):
        return len(self.input)
    
    def sample_data(self, batchsize):
        indices = np.random.choice(range(self.input.shape[0]), batchsize)
        return self.__getitem__(indices)
