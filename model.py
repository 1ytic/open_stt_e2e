import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def decrease_dim(x, layer, dim=1):
    if type(layer) != nn.modules.conv.Conv2d:
        return x
    p = layer.padding[dim]
    d = layer.dilation[dim]
    f = layer.kernel_size[dim]
    s = layer.stride[dim]
    x = (x + 2 * p - d * (f - 1) - 1) // s + 1
    return x


def is_time_decrease(layer):
    return decrease_dim(100, layer) != 100


class BatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        shape = list(x.size())
        x = x.view(-1, self.num_features)
        x = super().forward(x)
        shape = shape[:-1] + [self.num_features]
        x = x.view(shape)
        return x


class MaskConv(nn.Module):

    def __init__(self, layers):
        """
        Erase padding of the output based on the given lengths.
        Input needs to be in the shape of (NxCxDxT)
        :param layers: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.layers = layers

    def output_time(self, x):
        for layer in self.layers:
            x = decrease_dim(x, layer, dim=1)
        return x

    def output_dim(self, dim):
        channels = 0
        for layer in self.layers:
            dim = decrease_dim(dim, layer, dim=0)
            if type(layer) == nn.modules.conv.Conv2d:
                channels = layer.out_channels
        return dim * channels

    def forward(self, x, lengths):
        """
        :param x: The input of size NxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """

        mask = None

        for layer in self.layers:

            x = layer(x)

            if is_time_decrease(layer):

                lengths = decrease_dim(lengths, layer)

                n, c, d, t = x.size()

                mask = torch.zeros((n, 1, 1, t), dtype=torch.bool, device=x.device)

                for i, length in enumerate(lengths):
                    start = length.item()
                    length = t - start
                    if length > 0:
                        mask[i].narrow(2, start, length).fill_(1)

            if mask is not None:
                x = x.masked_fill(mask, 0)

        n, c, d, t = x.size()
        x = x.view(n, c * d, t)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # T x N x H

        return x, lengths


class AcousticModel(nn.Module):

    def __init__(self, input_size, hidden_size, prj_size, output_size,
                 n_layers=1, dropout=0, checkpoint=''):
        super(AcousticModel, self).__init__()
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(32, 32, kernel_size=(11, 11), stride=(2, 1), padding=(5, 5), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout(dropout)
        ))
        input_size = self.conv.output_dim(input_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True)
        self.prj = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(hidden_size, prj_size, bias=False))
        self.fc = nn.Sequential(BatchNorm1d(prj_size), nn.ReLU(inplace=True),
                                nn.Linear(prj_size, output_size))
        if len(checkpoint):
            print(checkpoint)
            self.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    def features(self, x, lengths):
        # Apply 2d convolutions
        x, lengths = self.conv(x, lengths)
        # Pack padded batch of sequences for RNN module
        x = pack_padded_sequence(x, lengths)
        # Forward pass through GRU
        x, _ = self.rnn(x)
        # Unpack padding
        x, _ = pad_packed_sequence(x)
        # Sum bidirectional GRU outputs
        x = x[:, :, :self.rnn.hidden_size] + x[:, :, self.rnn.hidden_size:]
        x = self.prj(x)
        return x, lengths

    def forward(self, x, lengths):
        x, lengths = self.features(x, lengths)
        x = self.fc(x)  # T x N x H
        x = log_softmax(x, dim=-1)
        return x, lengths


class LanguageModel(nn.Module):

    def __init__(self, emb_size, hidden_size, prj_size, vocab_size,
                 n_layers=1, dropout=0, blank=0, checkpoint=''):
        super(LanguageModel, self).__init__()
        # The gradient for blank input is always zero.
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=blank)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0)
        self.prj = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(hidden_size, prj_size, bias=False))
        self.fc = nn.Sequential(BatchNorm1d(prj_size), nn.ReLU(inplace=True),
                                nn.Linear(prj_size, vocab_size))
        if len(checkpoint):
            print(checkpoint)
            self.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    def features(self, x, lengths):
        init = torch.zeros((1, x.shape[1]), device=x.device).long()
        x = torch.cat([init, x.long()])
        x = self.emb(x)
        x = pack_padded_sequence(x, lengths + 1, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x)
        x = self.prj(x)
        return x

    def forward(self, x, lengths):
        x = self.features(x, lengths)
        x = self.fc(x)  # T x N x H
        return x

    def step_features(self, x, h=None):
        x = self.emb(x)
        x, h = self.rnn(x, h)
        x = self.prj(x)
        return x, h

    def step_forward(self, x, h=None):
        x, h = self.step_features(x, h)
        x = self.fc(x)  # T x N x H
        return x, h

    def step_init(self, batch_size):
        weight = next(self.rnn.parameters())
        return (weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))


class Transducer(nn.Module):

    def __init__(self, emb_size, vocab_size, hidden_size, prj_size,
                 am_layers=3, lm_layers=2, dropout=0, blank=0,
                 am_checkpoint='', lm_checkpoint=''):
        super(Transducer, self).__init__()

        self.blank = blank

        self.encoder = AcousticModel(40, hidden_size, prj_size, vocab_size,
                                     n_layers=am_layers, dropout=dropout,
                                     checkpoint=am_checkpoint)

        self.decoder = LanguageModel(emb_size, hidden_size, prj_size, vocab_size,
                                     n_layers=lm_layers, dropout=dropout, blank=blank,
                                     checkpoint=lm_checkpoint)

        for p in self.encoder.fc.parameters():
            p.requires_grads = False
        for p in self.decoder.fc.parameters():
            p.requires_grads = False

        self.fc = nn.Linear(prj_size, vocab_size)

    def joint(self, x, y):
        z = torch.tanh(x + y)
        z = self.fc(z)
        z = log_softmax(z, dim=-1)
        return z

    def forward(self, xs, ys, xn, yn):
        # encoder
        xs, xn = self.encoder.features(xs, xn)
        xs = xs.transpose(0, 1)
        # decoder
        zs = self.forward_decoder(xs, ys, yn)
        return zs, xs, xn

    def forward_decoder(self, xs, ys, yn):
        # decoder
        ys = self.decoder.features(ys, yn)
        ys = ys.transpose(0, 1)
        # align
        n, t, x_h = xs.size()
        n, u, y_h = ys.size()
        x = xs.unsqueeze(dim=2).expand(torch.Size([n, t, u, x_h]))
        y = ys.unsqueeze(dim=1).expand(torch.Size([n, t, u, y_h]))
        # predict
        zs = self.joint(x, y)
        return zs

    def greedy_decode(self, xs, sampled=False):

        n, t, h = xs.size()

        c = torch.zeros((1, n), device=xs.device).long()
        yd, (hd, cd) = self.decoder.step_features(c)

        s = torch.zeros((n, t), device=xs.device, dtype=torch.int)

        for i in range(t):

            z = self.joint(xs[:, i], yd[0])

            if sampled:
                c = torch.multinomial(z.exp(), num_samples=1).view(n)
            else:
                c = torch.argmax(z, dim=-1)

            s[:, i] = c

            c = c.view(1, n)

            mask = c == self.blank
            mask = mask.unsqueeze(-1)

            yd_next, (hd_next, cd_next) = self.decoder.step_features(c, (hd, cd))

            yd = torch.where(mask, yd, yd_next)
            hd = torch.where(mask, hd, hd_next)
            cd = torch.where(mask, cd, cd_next)

        return s
