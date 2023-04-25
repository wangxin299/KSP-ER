import torch
import torch.nn as nn
import math
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence


class SimpleRNN(nn.Module):
    """
    SimpleRNN  :use one-hot
    """

    def __init__(self, input_size=30, hidden_dim=200, num_skills=124, num_layers=1, dropout=0.7):
        super(SimpleRNN, self).__init__()
        self.MAX_NUM = 10000
        self.num_of_skills = num_skills
        self.embedding = nn.Embedding(self.MAX_NUM, int(math.log2(2*128)))  # math.log2(2*128) = 8

        self.rnn = nn.GRU(input_size=int(math.log2(2 * 128)), hidden_size=hidden_dim
                           , num_layers=num_layers, dropout=dropout, batch_first=True)
        # self.rnn = nn.LSTM(input_size=int(math.log2(2*128)), hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        # 即使RNN的输入数据是batch first，内部也会转为seq_len first。

        self.decoder1 = nn.Linear(hidden_dim, num_skills)

    def forward(self, x):
        '''
        :param x: [batch,seq]
        :return: probability [batch,seq]
        '''
        # print("seq_x", x.shape)
        # print("seq_x:",x)    # seq_x torch.Size([32, 8200])
        emb = self.embedding(x)
        # print("emb1", emb.shape)   # emb.shape torch.Size([32, 8200, 8])
        seq, _ = self.rnn(emb)
        # print("seq",seq.shape)  # seq.shape torch.Size([32, 8200, 200])
        seq = self.decoder1(seq)
        # print("seq", seq.shape)  # seq.shape torch.Size([32, 8200, 124]) seq -> [batch,seq,num_skills]

        return seq
