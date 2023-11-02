import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import trunc_normal_

# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    

# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim]+hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


# %%
class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


# %%
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(
            50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 frac=False,
                 attn=True,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=2048,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            # print(x_src.shape)
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src,
                                         src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)
            x_src = x_src.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            # print(x.shape, x_src.shape, hmask.shape)
            x = x.masked_fill(hmask == 0, 0)
            x_src = x_src.masked_fill(hmask ==0, 0)

        return x, x_src


# %%
class CrabNet(nn.Module):
    def __init__(self,
                 out_dims=256,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost',
                 pretrained=False):
        super().__init__()
        self.avg = True
        self.pretrained = pretrained
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device)
        if residual_nn == 'roost':
            # use the Roost residual network
            # self.out_hidden = [1024, 512, 256, 128]
            self.out_hidden = [1024, 512, 256]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
            self.res_hidden = [256,256,256]
            self.res_nn = ResidualNetwork(self.out_dims,
                                          self.out_dims,
                                          self.res_hidden)            
        else:
            # use a simpler residual network
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
        self.src_nn = torch.nn.Linear(self.d_model,self.out_dims)
        self.final = torch.nn.Linear(self.out_dims,1)
        self.final_trans = torch.nn.Linear(256, 1)
        self.input = torch.nn.Linear(257, 256)
        self.output = torch.nn.Linear(256, 256)
        # transformer
        # self.pos_encoder = PositionalEncoding(256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        en_layer_norm = nn.LayerNorm(256)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=en_layer_norm)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
        de_layer_norm = nn.LayerNorm(256)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6, norm=de_layer_norm)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, 256)))
        trunc_normal_(self.cls_token, std=0.02)
        # self.transformer = Transformer(d_model=256, batch_first=True)


    def forward(self, src, frac, trg, trg_attn_mask=None, trg_pad_mask=None):
        # with torch.no_grad():
        # trg = pad_sequence(trg, batch_first=True)
        # add start and end token tensors
        # trg = torch.cat([torch.zeros(trg.shape[0],trg.shape[1],1), trg], dim=2)
        # trg = torch.cat([torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device),\
        #                 trg, torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device)], dim=1)
        # trg[:,0,0] = 1; trg[:,-1,0] = -1  #trg shape is B*(len_seq+2)*(256+1)
        if trg_pad_mask is None: 
            trg_pad_mask = trg.eq(0).all(dim=-1).to(trg.device)
        if trg_attn_mask is None:
            trg_attn_mask = Transformer.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)
        output, x_src = self.encoder(src, frac)

        # average the "element contribution" at the end
        # mask so you only average "elements"
        src_pad_mask = (src == 0)
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        # print(mask.shape, src.shape)
        output = self.output_nn(output)  # simple linear
        src = (output + self.src_nn(x_src)).masked_fill(mask, 0)
        # print(src.shape, trg.shape)
        # cls_token = torch.zeros(src.shape[0], 1, src.shape[2]).to(src.device)
        # cls_token = trunc_normal_(cls_token, std=0.02)
        cls_tokens = self.cls_token.expand(src.shape[0], -1, -1)
        src = torch.cat([cls_tokens, src], dim=1)
        src_pad_mask = torch.cat([torch.zeros((src_pad_mask.shape[0], 1), dtype=torch.bool).to(src.device),\
                                   src_pad_mask], dim=1)

        memory = self.transformer_encoder(src, src_key_padding_mask = src_pad_mask,)
        out = self.transformer_decoder(trg, memory, tgt_mask=trg_attn_mask,\
                                       tgt_key_padding_mask = trg_pad_mask,)
        # out = self.transformer(src, self.input(trg), tgt_mask = trg_attn_mask, src_key_padding_mask = src_pad_mask,\
        #            tgt_key_padding_mask = trg_pad_mask,)
        
        output = out * (~trg_pad_mask).unsqueeze(dim=-1)
        # print(output.shape, trg_pad_mask.shape)
        output = self.final_trans(output.sum(dim=1)/ (~trg_pad_mask).sum(dim=1).unsqueeze(dim=1))
        return output, self.output(out)


# %%
if __name__ == '__main__':
    model = CrabNet()
