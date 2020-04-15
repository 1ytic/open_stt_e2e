import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

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

model = LanguageModel(128, 512, 256, len(labels), n_layers=3, dropout=0.3)
model.cuda()

bptt = 8
batch_size = 64

root = '/open-stt-e2e/data/'

train = [
    root + 'asr_public_phone_calls_1.csv',
    root + 'asr_public_phone_calls_2_aa.csv',
    root + 'asr_public_phone_calls_2_ab.csv',
    root + 'public_youtube1120_aa.csv',
    root + 'public_youtube1120_ab.csv',
    root + 'public_youtube1120_ac.csv',
    root + 'public_youtube1120_hq.csv',
    root + 'public_youtube700_aa.csv',
    root + 'public_youtube700_ab.csv'
]

test = [
    root + 'asr_calls_2_val.csv',
    root + 'buriy_audiobooks_2_val.csv',
    root + 'public_youtube700_val.csv'
]

train = TextDataset(train, labels, batch_size)
test = TextDataset(test, labels, batch_size)

test.shuffle(0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.99)

step = 0
writer = SummaryWriter(comment="_lm_bptt8_bs64_gn1_do0.3")

for epoch in range(1, 11):

    model.train()

    hidden = model.step_init(batch_size)

    err = AverageMeter('Loss/train')
    grd = AverageMeter('Gradient/train')

    train.shuffle(epoch)

    loader = DataLoaderCuda(train, batch_size=bptt, drop_last=True)

    for inputs, targets in loader:

        step += 1

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = detach_hidden(hidden)

        optimizer.zero_grad()

        output, hidden = model.step_forward(inputs, hidden)

        loss = criterion(output, targets.view(-1))
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        scheduler.step()

        err.update(loss.item())
        grd.update(grad_norm)

        writer.add_scalar(err.title + '/steps', loss.item(), step)
        writer.add_scalar(grd.title + '/steps', grad_norm, step)

        loader.set_description('Epoch %d %s %s' % (epoch, err, grd))

    model.eval()

    for i, lr in enumerate(scheduler.get_lr()):
        writer.add_scalar('LR/%d' % i, lr, epoch)

    err.summary(writer, epoch)
    grd.summary(writer, epoch)

    err = AverageMeter('Loss/test')

    loader = DataLoaderCuda(test, batch_size=bptt, drop_last=True)

    hidden = model.step_init(batch_size)

    with torch.no_grad():

        for inputs, targets in loader:

            output, hidden = model.step_forward(inputs, hidden)

            loss = criterion(output, targets.view(-1))

            err.update(loss.item())

            loader.set_description('Epoch %d %s' % (epoch, err))

    sys.stderr.write('\n')

    err.summary(writer, epoch)

    writer.flush()

    torch.save(model.state_dict(), writer.log_dir + '/model%d.bin' % epoch)

writer.close()

