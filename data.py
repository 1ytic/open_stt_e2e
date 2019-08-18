import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class Labels(object):

    def __init__(self):
        self.chars = ['<BLANK>', ' ', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def __call__(self, sentence):
        targets = []
        for c in sentence.strip().upper():
            targets.append(self.index[c])
        return targets

    def space(self):
        return self.index[' ']

    def blank(self):
        return self.index['<BLANK>']

    def string(self, targets, remove_repetitions=False):
        targets = list(targets.cpu().numpy())
        blank = self.blank()
        if remove_repetitions:
            last_t = -1
            unique = []
            for t in targets:
                if t != last_t:
                    last_t = t
                    unique.append(t)
            targets = unique
        sentence = [self.chars[t] for t in targets if t != blank]
        return ''.join(sentence)

    def is_accepted(self, sentence):
        sentence = sentence.strip().upper()
        if len(sentence) == 0:
            return False
        for c in sentence:
            if c not in self.index:
                return False
        return True

    def required_frames(self, sentence):
        targets = self(sentence)
        frames = len(targets)
        for t1, t2 in zip(targets[:-1], targets[1:]):
            if t1 == t2:
                frames += 1
        return frames


def load_data(source):
    if isinstance(source, list):
        data = [pd.read_csv(p, index_col='path', encoding='utf8') for p in source]
        return pd.concat(data)
    return pd.read_csv(source, index_col='path', encoding='utf8')


class TextDataset(Dataset):

    def __init__(self, source, labels, batch_size):
        data = load_data(source)
        b = labels.blank()
        self.utterances = [[b] + labels(t) for t in data['text'].values]
        self.data = []
        self.batch_size = batch_size

    def shuffle(self, epoch):
        np.random.RandomState(epoch).shuffle(self.utterances)
        data = np.concatenate(self.utterances)
        data = torch.tensor(data, dtype=torch.long)
        n = data.numel() // self.batch_size
        data = data.narrow(0, 0, n * self.batch_size)
        self.data = data.view(self.batch_size, -1).t()

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, i):
        return self.data[i], self.data[i+1]


class AudioDataset(Dataset):

    def __init__(self, source, labels):
        self.data = load_data(source)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        file = self.data.iloc[i]
        features = torch.tensor(np.load(file.name), dtype=torch.float32)
        targets = torch.tensor(self.labels(file['text']), dtype=torch.int)
        return features, targets

    def filter_by_length(self, length):
        self.filter(self.data[self.data['frames'] < length])

    def filter_by_conv(self, conv):
        frames = self.data['frames'].values
        frames = conv.output_time(frames)
        index = []
        for i, text in enumerate(self.data['text'].values):
            if self.labels.required_frames(text) <= frames[i]:
                index.append(i)
        self.filter(self.data.iloc[index])

    def filter(self, subset):
        diff = len(self.data) - len(subset)
        ratio = diff / len(self.data) * 100
        self.data = subset.sort_values('frames')
        print('filter %7d %7.2f%%' % (diff, ratio))


class BucketingSampler(Sampler):

    def __init__(self, data, batch_size=1):
        super(BucketingSampler, self).__init__(data)
        index = list(range(0, len(data)))
        self.bins = [index[i:i + batch_size] for i in range(0, len(index), batch_size)]

    def __iter__(self):
        for batch in self.bins:
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.RandomState(epoch).shuffle(self.bins)


def collate_fn(batch):

    batch = sorted(batch, key=lambda b: b[0].shape[0], reverse=True)

    n = len(batch)

    xs = []
    ys = []
    xn = torch.empty(n, dtype=torch.int)
    yn = torch.empty(n, dtype=torch.int)

    for i, (x, y) in enumerate(batch):
        xs.append(x)
        ys.append(y)
        xn[i] = len(x)
        yn[i] = len(y)

    # N x 1 x D x T
    xs = pad_sequence(xs, batch_first=True)
    xs = xs.unsqueeze(dim=1).transpose(2, 3).contiguous()

    return xs, ys, xn, yn


def collate_fn_ctc(batch):
    xs, ys, xn, yn = collate_fn(batch)
    ys = torch.cat(ys)
    return xs, ys, xn, yn


def collate_fn_rnnt(batch):
    xs, ys, xn, yn = collate_fn(batch)
    ys = pad_sequence(ys)
    return xs, ys, xn, yn


if __name__ == '__main__':

    labels = Labels()

    dataset = AudioDataset('/media/lytic/STORE/ru_open_stt_wav/public_youtube700_val.txt', labels)

    loader = DataLoader(dataset, num_workers=4, batch_size=32, collate_fn=collate_fn)

    x_lengths = []
    y_lengths = []

    for _, _, xn, yn in tqdm(loader):
        x_lengths.extend(list(xn.numpy()))
        y_lengths.extend(list(yn.numpy()))

    plt.title('x lengths')
    sns.distplot(x_lengths)
    plt.show()

    plt.title('y lengths')
    sns.distplot(y_lengths)
    plt.show()
