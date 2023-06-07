import torch.nn
from torch import nn
import torchbnn as bnn
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_bayesian(module, mean_init, bias_init, gain=1):
    mean_init(module.weight_mu.data, gain=gain)
    bias_init(module.bias_mu.data)
    return module

class DQN(nn.Module):
    def __init__(self,input_dim, output_dim, nn_type:str, mu:float=0,sigma:float=0.1):
        super().__init__()
        self.mu=mu
        self.sigma=sigma
        self.hidden_size=64
        input_shape=np.array(input_dim)[0].shape[0]           #input_dim ~= observation_space
        output_shape=np.array(output_dim)[0].n     #output_dim ~= activation_space
        _init1 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x,0), np.sqrt(2))
        _init2 = lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x,0), gain=0.01 )


        _init1_bayesian = lambda m: init_bayesian(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x,0), np.sqrt(2))
        _init2_bayesian = lambda m: init_bayesian(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x,0), gain=0.01 )

        if nn_type == "NN":
            self.layer1=_init1(nn.Linear(in_features=input_shape,out_features=self.hidden_size) )#100
            self.layer2=_init1(nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size))
            self.layer3=_init2(nn.Linear(in_features=self.hidden_size,out_features=output_shape))
            #
            # self.layer1 = nn.Linear(in_features=input_shape, out_features=self.hidden_size)  # 100
            # self.layer2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            # self.layer3 = nn.Linear(in_features=self.hidden_size, out_features=output_shape)
        elif nn_type=="BNN":
            self.layer1 = _init1_bayesian(bnn.BayesLinear(prior_mu=self.mu,prior_sigma=self.sigma,in_features=input_shape, out_features=self.hidden_size))
            self.layer2 = _init1_bayesian(bnn.BayesLinear(prior_mu=self.mu,prior_sigma=self.sigma,in_features=self.hidden_size, out_features=self.hidden_size))
            self.layer3 = _init2_bayesian(bnn.BayesLinear(prior_mu=self.mu,prior_sigma=self.sigma,in_features=self.hidden_size, out_features=output_shape))
        else:
            raise Exception('Incorrect nn_type')

    def forward(self,state):
        x = self.layer1(state)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

    def get_max(self,x:np.ndarray):
        x=x.argmax()
        return x
