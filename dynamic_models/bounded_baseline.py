import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import torchify
from simulator.simulator_data_gen import fLimits_vec, gLimits_vec, fBaselineLimits_vec
from simulator.simulator_constants import dt, J



class DynamicsModelBoundedBaseline(nn.Module):
    '''
    With bounds, without decomposition.
    '''
    def __init__(self, input_dim=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), output_dim=1):
        """
        Dim of [x_t, w_t, u_t, f_U, f_L, g_U, g_L, fB_U, fB_L] = n, n, m, n, n, n*m, n*m, n, n
        output_dim: [w_t+1]: n
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f_model_input_dim = self.input_dim[:5].sum()
        self.f_model_output_dim = self.output_dim
        self.f_model = nn.Sequential(
          nn.Linear(self.f_model_input_dim, 64),
          nn.Sigmoid(),
          nn.Linear(64, 64),
          nn.Tanh(),
          nn.Linear(64, 64),
          nn.Tanh(),
          nn.Linear(64, self.f_model_output_dim)
        )
        

    def forward(self, x):
        '''Forward pass'''
        f_output = self.f_model(torch.cat((x[:, :int(self.input_dim[:3].sum())], x[:, int(self.input_dim[:-2].sum()):int(self.input_dim.sum())]), dim=1))
        fB_U = torch.tensor(x[:, self.input_dim[:-2].sum(): self.input_dim[:-1].sum()], device=self.device)
        fB_L = torch.tensor(x[:, self.input_dim[:-1].sum(): self.input_dim.sum()], device=self.device)
        f_part = (fB_U - fB_L) / 2 * F.tanh(f_output) + (fB_U + fB_L) / 2
        
        return f_part

    
    def get_prediction(self, obs, acs):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :return: a numpy array of the predicted next states (s_t+1)
        """
        fL_k, fU_k = fLimits_vec(obs[..., 0], obs[..., 1])
        gL_k, gU_k = gLimits_vec(obs[..., 0], obs[..., 1])
        fL_k, fU_k = fBaselineLimits_vec(obs[..., 0], obs[..., 1])
        x = torchify(np.vstack([obs[..., 0], obs[..., 1], acs.squeeze(), fU_k, fL_k, gU_k, gL_k, fU_k, fL_k]).T, device=self.device)
        prediction = self(x)
        w_plus = prediction.detach().cpu().numpy()
        x_plus = obs[..., [0]] + obs[..., [1]] * dt
        return np.hstack([x_plus, w_plus])
    
