import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from data import Labels, AudioDataset, DataLoaderCuda, collate_audio, BucketingSampler

from model import Transducer
from utils import AverageMeter

from warp_rnnt import rnnt_loss
from pytorch_edit_distance import remove_blank, AverageWER, AverageCER


torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
np.random.seed(1)

labels = Labels()

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.4,
                   am_checkpoint='exp/am.bin', lm_checkpoint='exp/lm.bin')
model.cuda()

train = [
    '/media/lytic/STORE/ru_open_stt_wav/public_youtube1120_hq.txt',
    #'/media/lytic/STORE/ru_open_stt_wav/public_youtube700_aa.txt'
]

test = [
    '/media/lytic/STORE/ru_open_stt_wav/asr_calls_2_val.txt',
    '/media/lytic/STORE/ru_open_stt_wav/buriy_audiobooks_2_val.txt',
    '/media/lytic/STORE/ru_open_stt_wav/public_youtube700_val.txt'
]

train = AudioDataset(train, labels)
test = AudioDataset(test, labels)

train.filter_by_conv(model.am.conv)
train.filter_by_length(400)

test.filter_by_conv(model.am.conv)
test.filter_by_length(500)

parameters = [
    {"params": model.am.parameters(), "lr": 3e-5},
    {"params": model.lm.parameters(), "lr": 3e-5},
    {"params": model.fc.parameters(), "lr": 3e-4}
]

optimizer = torch.optim.Adam(parameters, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

sampler = BucketingSampler(train, 32)

train = DataLoaderCuda(train, collate_fn=collate_audio, batch_sampler=sampler)
test = DataLoaderCuda(test, collate_fn=collate_audio, batch_size=16)

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

for epoch in range(10):

    sampler.shuffle(epoch)

    model.train()

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')

    for xs, ys, xn, yn in train:

        optimizer.zero_grad()

        zs, xs, xn = model(xs, ys.t(), xn, yn)

        loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 50)

        optimizer.step()
        scheduler.step()

        err.update(loss.item())
        grd.update(grad_norm)

        train.set_description('epoch %d %s %s' % (epoch + 1, err, grd))

    model.eval()

    err = AverageMeter('loss')
    cer = AverageCER(blank, space)
    wer = AverageWER(blank, space)

    with torch.no_grad():

        for xs, ys, xn, yn in test:

            zs, xs, xn = model(xs, ys.t(), xn, yn)

            loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

            xs = model.greedy_decode(xs)

            err.update(loss.item())

            remove_blank(xs, xn, blank)

            wer.update(xs, ys, xn, yn)
            cer.update(xs, ys, xn, yn)

            test.set_description('epoch %d %s %s %s' % (epoch + 1, err, cer, wer))

    sys.stderr.write('\n')

    torch.save(model.state_dict(), 'exp/asr.bin')
