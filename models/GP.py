from __future__ import division
import os
import numbers
from typing import Optional

from torch.nn import Parameter
from .DCI_Block import DCI_Block
from .FD import FDNet
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

#用於圖形建立
class Linear(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._mlp(X)

# WaveForm 做圖學習使用的 他是用一開始 GraphConstructor 建立的圖
class MixProp(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:

        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            
            H = self._alpha * X + (1 - self._alpha) * torch.einsum(
                "ncwl,vw->ncvl", (H, A)
            )
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0
class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()
    
# 我們做圖學習使用的 會建立兩個方向的圖來做 由於是利用輸入分解後的數據來做的圖 每一層進來的資訊量會不一樣
# 圖形建立是用輸入分解的資料建立的 不是用GraphConstructor
class Dy_MixProp(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(Dy_MixProp, self).__init__()
        self.nconv = dy_nconv()
        self._mlp1 = Linear((gdep + 1) * c_in, c_out)
        self._mlp2 = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha
        
        self.lin1 = Linear(c_in,c_in)
        self.lin2 = Linear(c_in,c_in)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    # 雖然有輸入 GraphConstructor的 A 但沒有使用 是用X這些輸入的資料建立的
    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        x1 = torch.tanh(self.lin1(X))
        x2 = torch.tanh(self.lin2(X))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)
        h = X
        out = [h]
        for i in range(self._gdep):
            h = self._alpha*X + (1-self._alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self._mlp1(ho)

        h = X
        out = [h]
        for i in range(self._gdep):
            h = self._alpha * X + (1 - self._alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self._mlp2(ho)

        return ho1+ho2

#做 Dilated 也就是 TCN 學習的部份 會出現在過去的MTCGNN 跟 WaveForM 的概念 此處研用於模型的其中一塊
class DilatedInception(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        
        
        Y = [0 ,0 ,0, 0]
        for i in range(len(self._kernel_set)):
            Y[i] = X[i].permute(0, 2, 1, 3)
        
        Y = torch.cat(Y, dim=2)
        
        X = Y.permute(0, 2, 1, 3)
        return  X



class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("_weight", None)
            self.register_parameter("_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        if self._elementwise_affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps,
            )
        else:
            return F.layer_norm(
                X, tuple(X.shape[1:]), self._weight, self._bias, self._eps
            )

#整個 GPLayer 學習組成 裡面包含 DCI 跟 Graph 等學習 基於 WaveForm 改良
class GPModuleLayer(nn.Module):

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        hiddenDCI: int,
        propalpha: float,
    ):
        super(GPModuleLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true
        
        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)
        #如果要改 DCI 參數在這裡改
        self.scinet = DCI_Block( 
                #序列長
                output_len=32,
                input_len=32,
                #solar
                # input_dim=321,
                # input_dim=21,
                # input_dim=8,
                # input_dim=7,
                #特徵數
                input_dim=num_nodes,
                # hid_size=4,
                # num_stacks=1,
                # num_levels=2,
                num_stacks=1,
                # hid_size=1, #traffic
                # hid_size=8 ,#ELC
                # hid_size=16 #Weather solar
                hid_size=hiddenDCI ,
                num_levels=1,
                num_decoder_layer=1,
                concat_len=0,
                groups=1,
                kernel=5 ,
                dropout=0,
                single_step_output_One=0,
                positionalE=False,
                modified=True,
                RIN=False
        )

        # self.TLN = FFT_Conv_Net(channels=32,
        #         pred_num=32,
        #         F_channels=321,
        # )
        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )
        
        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        # 定義 GCN 使用的部份 
        if gcn_true:
            self._mixprop_conv1 = Dy_MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

            self._mixprop_conv2 =  Dy_MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:

        X_residual = X

        X1 = X[..., :1]
        X1 = X1.squeeze(-1)
        y= self.scinet(X1)
        X_tmp = self._gate_conv(X)
        X_tmp = torch.sigmoid(X_tmp)
        X_scinet_t = torch.tanh(y)
        X_scinet_s = torch.sigmoid(y)
        X_scinet_t = X_scinet_t.unsqueeze(-1)  
        X_scinet_t = X_scinet_t.expand(-1, -1, -1, X_tmp.size(3))  
        X_scinet_s = X_scinet_s.unsqueeze(-1)  
        X_scinet_s = X_scinet_s.expand(-1, -1, -1, X_tmp.size(3)) 
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)

        # 由 DCI + Dilated 組成 各部份由filte + Gate 而成
        X = X_filter * X_gate *X_scinet_t *X_scinet_s
        # X.shape 是 torch.Size([batch, 32, node, factor])
        # X = X_filter * X_gate 
      

        # 學習完 DCI 跟 Dilate 後處李 GCN 使用的部份 
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
            
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class GPModule(nn.Module):
    
    def __init__(
        self,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        dropout: float,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        layers: int,
        propalpha: float,
        hiddenDCI : int,
        layer_norm_affline: bool,
        graph_constructor,
        xd: Optional[int] = None,
    ):
        super(GPModule, self).__init__()
        
        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)
        self.hiddenDCI  = hiddenDCI
        self._gp_layers = nn.ModuleList()
        
        self._graph_constructor = graph_constructor
        
        self._set_receptive_field(dilation_exponential, kernel_size, layers)
        # 初始化各個參數
        new_dilation = 1
        for j in range(1, layers + 1):
            self._gp_layers.append(
                GPModuleLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    hiddenDCI  = self.hiddenDCI ,
                    
                    propalpha=propalpha,

                )
            )
            
            new_dilation *= dilation_exponential
        
        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )
        
        self._reset_parameters()
        
    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):
    
        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        
        if self._seq_length > self._receptive_field:
            
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )
            
            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )
            
        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )
            
            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )
        
        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        
        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1
    
    def forward(
        self,
        context: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        
        X_in = context.permute(0, 3, 2, 1)
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."
        
        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )
        
        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)
        
        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for gp in self._gp_layers:
                
                X, X_skip = gp(X, X_skip, A_tilde, self._idx.to(X_in.device), self.training)
        else:
            for gp in self._gp_layers:
                X, X_skip = gp(X, X_skip, A_tilde, idx, self.training)
        
        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        
        return X
    
    
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)

