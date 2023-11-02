import torch
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence

src = [torch.rand((2,256)), torch.rand((3,256)), torch.rand((4,256)),]
src = pad_sequence(src, batch_first=True)
src_mask = src.eq(0).all(dim=-1)
print(src.shape, src_mask)


trg = [torch.rand((6,256)), torch.rand((12,256)), torch.rand((4,256)),]
trg = pad_sequence(trg, batch_first=True)
# add start and end token tensors
# trg = torch.cat([torch.zeros(trg.shape[0],trg.shape[1],1), trg], dim=2)
trg = torch.cat([torch.ones(trg.shape[0],1,trg.shape[2]),\
                 trg, torch.ones(trg.shape[0],1,trg.shape[2])], dim=1)
trg = torch.cat([torch.zeros(trg.shape[0],trg.shape[1],1), trg], dim=2)
trg[:,0,0] = 1; trg[:,-1,0] = -1
trg_mask = trg.eq(0).all(dim=-1)
trg_attn_mask = Transformer.generate_square_subsequent_mask(trg.shape[1])
print(trg.shape, trg_mask, trg_attn_mask)


transformer = Transformer(d_model=256, batch_first=True)
out = transformer(src, trg, tgt_mask = trg_attn_mask, src_key_padding_mask = src_mask,\
                   tgt_key_padding_mask = trg_mask,)
print(out.shape, sum(p.numel() for p in transformer.parameters()))