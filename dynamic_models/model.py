import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import torchify
from simulator.simulator_data_gen import fLimits_vec, gLimits_vec, fBaselineLimits_vec
from simulator.simulator_constants import dt, J
  
class DynamicsModel(nn.Module):
    '''
    With bounds, with decomposition.
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
        self.f_model_input_dim = self.input_dim[0] + self.input_dim[1] + self.input_dim[3] + self.input_dim[4]
        self.f_model_output_dim = self.output_dim
        self.g_model_input_dim = self.input_dim[0] + self.input_dim[1] + self.input_dim[5] + self.input_dim[6]
        self.g_model_output_dim = self.input_dim[-3]
        self.f_model = nn.Sequential(
          nn.Linear(self.f_model_input_dim, 64),
          nn.Sigmoid(),
          nn.Linear(64, 64),
          nn.Tanh(),
          nn.Linear(64, 64),
          nn.Tanh(),
          nn.Linear(64, self.f_model_output_dim)
        )

        self.g_model = nn.Sequential(
          nn.Linear(self.g_model_input_dim, 64),
          nn.Sigmoid(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, self.g_model_output_dim)
        )
        self.g_part_true = torch.tensor([dt/J])

    def forward(self, x):
        '''Forward pass'''
        f_output = self.f_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:3].sum()):int(self.input_dim[:5].sum())]), dim=1))
        f_U = x[:, self.input_dim[:3].sum(): self.input_dim[:4].sum()]
        f_L = x[:, self.input_dim[:4].sum(): self.input_dim[:5].sum()]
        f_part = (f_U - f_L) / 2 * F.tanh(f_output) + (f_U + f_L) / 2
        # print(f_U, f_L)
        g_output = self.g_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:5].sum()):int(self.input_dim[:7].sum())]), dim=1))
        g_U = x[:, self.input_dim[:-4].sum(): self.input_dim[:-3].sum()]
        g_L = x[:, self.input_dim[:-3].sum(): self.input_dim[:-2].sum()]
        
        g_part = (g_U - g_L) / 2 * F.tanh(g_output) + (g_U + g_L) / 2
        

        g_matrix = g_part.reshape(-1, self.input_dim[0], self.input_dim[2])
        u_t = x[:, self.input_dim[:2].sum() : self.input_dim[:3].sum()]
        
        g_u_part = torch.einsum('bij,bj->bi', g_matrix, u_t)
        return f_part + g_u_part

    def forward_f(self, x):
        f_output = self.f_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:3].sum()):int(self.input_dim[:5].sum())]), dim=1).to(self.device))
        f_U = torch.tensor(x[:, self.input_dim[:-6].sum(): self.input_dim[:-5].sum()], device=self.device)
        f_L = torch.tensor(x[:, self.input_dim[:-5].sum(): self.input_dim[:-4].sum()], device=self.device)
        
        f_part = (f_U - f_L) / 2 * F.tanh(f_output) + (f_U + f_L) / 2
        
        return f_part

    def forward_g(self, x):
        g_output = self.g_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:5].sum()):int(self.input_dim[:7].sum())]), dim=1).to(self.device))
        g_U = torch.tensor(x[:, self.input_dim[:-4].sum(): self.input_dim[:-3].sum()], device=self.device)
        g_L = torch.tensor(x[:, self.input_dim[:-3].sum(): self.input_dim[:-2].sum()], device=self.device)
        
        g_part = (g_U - g_L) / 2 * F.tanh(g_output) + (g_U + g_L) / 2
        
        return g_part

    def forward_g_output(self, x):
        g_output = self.g_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:5].sum()):int(self.input_dim.sum())]), dim=1).to(self.device))
        
        return g_output
    
    def forward_g_bar(self, x):
        g_output = self.g_model(torch.cat((x[:, :int(self.input_dim[:2].sum())], x[:, int(self.input_dim[:5].sum()):int(self.input_dim[:7].sum())]), dim=1).to(self.device))
        g_U = torch.tensor(x[:, self.input_dim[:-4].sum(): self.input_dim[:-3].sum()], device=self.device)
        g_L = torch.tensor(x[:, self.input_dim[:-3].sum(): self.input_dim[:-2].sum()], device=self.device)
        
        g_part = (g_U - g_L) / 2 * F.tanh(g_output) + (g_U + g_L) / 2
        
        return (g_U + g_L) / 2

    def get_prediction(self, obs, acs):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :return: a numpy array of the predicted next states (s_t+1)
        """
        
        fL_k, fU_k = fLimits_vec(obs[..., 0], obs[..., 1])
        gL_k, gU_k = gLimits_vec(obs[..., 0], obs[..., 1])
        x = torchify(np.vstack([obs[..., 0], obs[..., 1], acs.squeeze(), fU_k, fL_k, gU_k, gL_k]).T, device=self.device)
        prediction = self(x)
        w_plus = prediction.detach().cpu().numpy()
        x_plus = obs[..., [0]] + obs[..., [1]] * dt
        return np.hstack([x_plus, w_plus])