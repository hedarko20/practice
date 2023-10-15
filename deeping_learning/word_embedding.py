import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

batch_shape = 10
max_len = 50
dic_len = 300
embedding_dim = 20

# 随机生成batch_shape 大小，最大长度为 max_len 的句子(如果是正常的句子需要自己创建词表并索引化)
src_sen_len = torch.randint(1, max_len, (batch_shape, ))
tgt_sen_len = torch.randint(1, max_len, (batch_shape, )) 
print(src_sen_len.shape)
print(tgt_sen_len.shape)
src_sen = [F.pad(torch.randint(1, dic_len, (L, )), (0, max_len-L)) for L in src_sen_len]
tgt_sen = [F.pad(torch.randint(1, dic_len, (L, )), (0, max_len-L)) for L in tgt_sen_len]
src_sen = torch.stack(src_sen)
tgt_sen = torch.stack(tgt_sen)
print(src_sen.shape)
print(tgt_sen.shape)
# embedding
embedding_layer = nn.Embedding(dic_len+1, embedding_dim)
ebd_src_sen = embedding_layer(src_sen)
ebd_tgt_sen = embedding_layer(tgt_sen)
print(ebd_src_sen)
print(ebd_tgt_sen)
print(ebd_src_sen.shape)
print(ebd_tgt_sen.shape)
