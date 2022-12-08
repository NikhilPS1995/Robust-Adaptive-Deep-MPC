import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import torchify
from simulator.simulator_data_gen import fLimits_vec, gLimits_vec, fBaselineLimits_vec
from simulator.simulator_constants import dt, J


class DynamicsModelBaseline(nn.Module):
    '''
    Without bounds, without decomposition.
    '''
    def __init__(self, input_dim=np.array([1, 1, 1]), output_dim=1):
        """
        Dim of [x_t, w_t, u_t, f_U, f_L, g_U, g_L, fB_U, fB_L] = n, n, m, n, n, n*m, n*m, n, n
        output_dim: [w_t+1]: n
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.f_model_input_dim = self.input_dim[0] + self.input_dim[1] + self.input_dim[2]
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
        
        f_output = self.f_model(x[:, :int(self.input_dim[:3].sum())])
        return f_output
    
    def get_prediction(self, obs, acs):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)        
        :return: a numpy array of the predicted next states (s_t+1)
        """
        x = torchify(np.hstack([obs, acs]), device=self.device)
        prediction = self(x)
        w_plus = prediction.detach().cpu().numpy()
        x_plus = obs[..., [0]] + obs[..., [1]] * dt
        return np.hstack([x_plus, w_plus])
    
