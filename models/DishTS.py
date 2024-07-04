import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    def __init__(self, args):
        super().__init__()
        init = 'uniform' #'standard', 'avg' or 'uniform' etc.
        activate = True
        # n_series = 321 # number of series
        # n_series = 21 # number of series
        # n_series = 8 # number of series
        # n_series = 7 # number of series
        n_series = args.n_points # number of series
        # lookback = 96 # lookback length
        lookback = args.seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        # elif init == 'zeros':
        #     self.reduce_mlayer = nn.Parameter(torch.zeros(n_series, lookback, 2))
        # elif init == 'normal':
        #     self.reduce_mlayer = nn.Parameter(torch.randn(n_series, lookback, 2)/lookback)
        # elif init == 'xavier':
        #     self.reduce_mlayer = nn.Parameter(torch.Tensor(n_series, lookback, 2))
        #     nn.init.xavier_uniform_(self.reduce_mlayer)
        # elif init == 'he':
        #     self.reduce_mlayer = nn.Parameter(torch.Tensor(n_series, lookback, 2))
        #     nn.init.kaiming_uniform_(self.reduce_mlayer, a=0, mode='fan_in', nonlinearity='relu')

        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward'):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            # dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1) 
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih
