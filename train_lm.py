import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from data import Labels, TextDataset, DataLoaderCuda

from model import LanguageModel
from utils import AverageMeter


def detach_hidden(h):
    """Detach hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(detach_hidden(v) for v in h)


torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

labels = Labels()
num_labels = len(labels)

model = LanguageModel(128, 512, 256, num_labels, n_layers=3, dropout=0.3)
model.cuda()

bptt = 8
batch_size = 32

train = [
    '/media/lytic/STORE/ru_open_stt_wav/text/public_youtube1120_hq.txt',
    '/media/lytic/STORE/ru_open_stt_wav/text/public_youtube1120.txt',
    '/media/lytic/STORE/ru_open_stt_wav/text/public_youtube700.txt'
]

test = [
    '/media/lytic/STORE/ru_open_stt_wav/text/asr_calls_2_val.txt',
    '/media/lytic/STORE/ru_open_stt_wav/text/buriy_audiobooks_2_val.txt',
    '/media/lytic/STORE/ru_open_stt_wav/text/public_youtube700_val.txt'
]

train = TextDataset(train, labels, batch_size)
test = TextDataset(test, labels, batch_size)

test.shuffle(0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.99)

for epoch in range(20):

    model.train()

    hidden = model.step_init(batch_size)

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')

    train.shuffle(epoch)

    loader = DataLoaderCuda(train, batch_size=bptt, drop_last=True)

    for inputs, targets in loader:

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = detach_hidden(hidden)

        optimizer.zero_grad()

        output, hidden = model.step_forward(inputs, hidden)

        loss = criterion(output.view(-1, num_labels), targets.view(-1))
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

        err.update(loss.item())
        grd.update(grad_norm)

        lr = scheduler.get_lr()[0]

        loader.set_description('epoch %d lr %.6f %s %s' % (epoch + 1, lr, err, grd))

    model.eval()

    err = AverageMeter('loss')

    loader = DataLoaderCuda(test, batch_size=bptt, drop_last=True)

    hidden = model.step_init(batch_size)

    with torch.no_grad():

        for inputs, targets in loader:

            output, hidden = model.step_forward(inputs, hidden)

            loss = criterion(output.view(-1, num_labels), targets.view(-1))

            err.update(loss.item())

            loader.set_description('epoch %d %s' % (epoch + 1, err))

    sys.stderr.write('\n')

    torch.save(model.state_dict(), 'exp/lm.bin')
