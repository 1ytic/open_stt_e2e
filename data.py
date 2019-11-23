import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence


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

    def __init__(self, source, labels, model=None, length=0):
        self.data = load_data(source)
        self.labels = labels
        if model is not None:
            self.filter_by_model(model)
        if length > 0:
            self.filter_by_length(length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        file = self.data.iloc[i]
        features = torch.tensor(np.load(file.name), dtype=torch.float32)
        targets = torch.tensor(self.labels(file['text']), dtype=torch.int)
        return features, targets

    def filter_by_length(self, length):
        self.filter(self.data[self.data['frames'] < length])

    def filter_by_model(self, model):
        frames = self.data['frames'].values
        frames = model.output_time(frames)
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

    def __init__(self, data, size=1, limit=sys.maxsize):
        super().__init__(data)
        index = list(range(len(data)))
        self.bins = [index[i:i + size] for i in range(0, len(index), size)]
        self.limit = limit

    def __iter__(self):
        for batch in self.bins[:self.limit]:
            yield batch

    def __len__(self):
        return len(self.bins[:self.limit])

    def shuffle(self, epoch):
        np.random.RandomState(epoch).shuffle(self.bins)


def collate_audio(batch):

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
    xs = xs.unsqueeze(dim=1).transpose(2, 3)

    # N x S
    ys = pad_sequence(ys, batch_first=True)

    return xs, ys, xn, yn


class DataLoaderCuda(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['num_workers'] = 4
        kwargs['pin_memory'] = True
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.progress = tqdm(super().__iter__())
        for cpu in self.progress:
            gpu = []
            for values in cpu:
                gpu.append(values.cuda(non_blocking=True))
            yield gpu

    def shuffle(self, epoch):
        self.batch_sampler.shuffle(epoch)

    def set_description(self, desc):
        self.progress.set_description(desc)

    def close(self):
        self.progress.close()


def split_train_dev_test(root, labels, model, batch_size=32):

    train = [
        root + '/asr_public_phone_calls_1.csv',
        root + '/public_youtube1120_hq.csv',
        root + '/public_youtube700_aa.csv',
        root + '/public_youtube700_ab.csv'
    ]

    dev = [
        root + '/asr_public_phone_calls_1.csv',
        root + '/public_youtube1120_hq.csv',
    ]

    test = [
        root + '/asr_calls_2_val.csv',
        root + '/buriy_audiobooks_2_val.csv',
        root + '/public_youtube700_val.csv'
    ]

    train = AudioDataset(train, labels, model, 400)
    dev = AudioDataset(dev, labels, model, 1000)
    test = AudioDataset(test, labels, model, 1000)

    sampler1 = BucketingSampler(train, size=batch_size)
    sampler2 = BucketingSampler(dev, size=1, limit=1000)

    sampler2.shuffle(0)

    train = DataLoaderCuda(train, collate_fn=collate_audio, batch_sampler=sampler1)
    dev = DataLoaderCuda(dev, collate_fn=collate_audio, batch_sampler=sampler2)
    test = DataLoaderCuda(test, collate_fn=collate_audio, batch_size=16)

    return train, dev, test


if __name__ == '__main__':

    labels = Labels()

    dataset = AudioDataset('ru_open_stt_wav/public_youtube700_val.csv', labels)

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_audio)

    x_lengths = []
    y_lengths = []

    for _, _, xn, yn in tqdm(loader):
        x_lengths.extend(list(xn.numpy()))
        y_lengths.extend(list(yn.numpy()))

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.title('x lengths')
    sns.distplot(x_lengths)
    plt.show()

    plt.title('y lengths')
    sns.distplot(y_lengths)
    plt.show()
