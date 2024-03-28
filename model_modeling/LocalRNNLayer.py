import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout=0.2):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize - 1, 37152, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.ksize - 1, input_dim)).cuda()

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x)  # b x seq_len x ksize x d_model 
        batch, l, ksize, d_model = x.shape
        gg = x.view(-1, self.ksize, d_model)  
        h1 = self.rnn(gg)
        kk = h1[0]  
        h = kk[:, -1, :]  
        return h.view(batch, l, d_model) 

    def get_K(self, x):
        batch_size, l, d_model = x.shape 
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)  
        x = torch.cat((zeros, x), dim=1) 
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])  
        key = key.reshape(batch_size, l, self.ksize, -1) 
        return key


class LocalRNNLayer(nn.Module):
    def __init__(self, config):
        super(LocalRNNLayer, self).__init__()
        input_dim = config.input_dim
        output_dim = config.output_dim
        rnn_type = config.rnn_type
        ksize = config.ksize
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout=0.2)
        self.connection = SublayerConnection(output_dim, dropout=0.2)

    def forward(self, x):
        x = self.connection(x, self.local_rnn)
        return x
