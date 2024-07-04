import torch.nn as nn

from models.ConvBlock import ConvBlock
from models.embed import DataEmbedding


class Decomposed_block(nn.Module):
    def __init__(self, enc_in, d_model, seq_kernel, dropout, attn_nums, ICOM, h_nums, label_len, pred_len, timebed):
        super(Decomposed_block, self).__init__()
        self.enc_in = enc_in - timebed
        self.pred_len = pred_len
        self.embed = DataEmbedding(d_model, dropout)
        self.ICOM = ICOM
      
        pro_conv2d = [ConvBlock(d_model, d_model, seq_kernel, ICOM=False, pool=False, dropout=dropout)
                          for _ in range(attn_nums)]
        self.pro_conv2d = nn.ModuleList(pro_conv2d)

        self.F = nn.Flatten(start_dim=2)
        final_len = label_len // (2 ** h_nums) if ICOM else label_len
        self.FC = nn.Linear(final_len * d_model, pred_len)

        self.timebed = timebed
        if self.timebed:
            self.time_layer = nn.Linear(self.timebed, self.enc_in * d_model)

    def forward(self, x):
    
        x = self.embed(x)
        x_2d = x.clone()
        for conv2d in self.pro_conv2d:
            x_2d = conv2d(x_2d)
        # print('x_2d:',x_2d.shape)
        x_2d_out = self.F(x_2d.transpose(1, -1))
        # print('x_2d_out:',x_2d_out.shape)
        x_out = self.FC(x_2d_out).transpose(1, 2)
        # print('x_out:',x_out.shape)
        return x_out


class FDNet(nn.Module):
    def __init__(self, enc_in, c_out, label_len, pred_len,
                 seq_kernel=3, attn_nums=3, timebed='hour',
                #  d_model=64, pyramid=4, ICOM=False, dropout=0.0):
                d_model=64, pyramid=4, ICOM=False, dropout=0.0):
        super(FDNet, self).__init__()
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        timebed = int(type_bed[timebed])
        self.enc_in = enc_in
        self.timebed = timebed
        self.pyramid = pyramid

        self.label_len = label_len
        self.pred_len = pred_len
        self.c_out = c_out
        self.d_model = d_model
        FDNet_blocks = [Decomposed_block(enc_in, d_model, seq_kernel, dropout, attn_nums, ICOM, 1,
                                         label_len // (2 ** pyramid), pred_len, self.timebed)] + \
                       [Decomposed_block(enc_in, d_model, seq_kernel, dropout, attn_nums - i, ICOM, i + 1,
                                         label_len // (2 ** (pyramid - i)), pred_len, self.timebed)
                        for i in range(pyramid + 1)]
        self.FDNet_blocks = nn.ModuleList(FDNet_blocks)

    def forward(self, x_enc):
        # print('x_enc:',x_enc.shape)
        enc_input = x_enc[:, :self.label_len, :]
        # print('enc_input:',enc_input.shape)
        enc_input_list = [enc_input[:, -self.label_len // (2 ** self.pyramid):, :]]
        enc_out = 0
        num_output = 0
        for i in range(self.pyramid):
            enc_input_list.append(enc_input[:, -self.label_len // (2 ** (self.pyramid - i - 1)):
                                               -self.label_len // (2 ** (self.pyramid - i)), :])
        for curr_input, FD_b in zip(enc_input_list, self.FDNet_blocks):
            transformed = FD_b(curr_input.unsqueeze(-1))
            enc_out += transformed
            num_output += 1

        return enc_out / num_output
