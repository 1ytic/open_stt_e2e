import sys
import torch
import torch.nn as nn

import numpy as np

from data import Labels, AudioDataset, DataLoader, collate_fn_rnnt, BucketingSampler

from tqdm import tqdm

from model import Transducer
from utils import AverageMeter, entropy

import decoder

from warp_rnnt import rnnt_loss

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

labels = Labels()

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.3,
                   am_checkpoint='exp/am.bin', lm_checkpoint='exp/lm.bin')

train = AudioDataset('/media/lytic/STORE/ru_open_stt_wav/public_youtube1120_hq.txt', labels)
test = AudioDataset('/media/lytic/STORE/ru_open_stt_wav/public_youtube700_val.txt', labels)

train.filter_by_conv(model.encoder.conv)
train.filter_by_length(400)

test.filter_by_conv(model.encoder.conv)
test.filter_by_length(200)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

model.cuda()

sampler = BucketingSampler(train, 32)

train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_sampler=sampler)
test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_size=16)

for epoch in range(10):

    sampler.shuffle(epoch)

    model.train()

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')

    progress = tqdm(train)
    for xs, ys, xn, yn in progress:

        optimizer.zero_grad()

        xs = xs.cuda(non_blocking=True)
        ys = ys.cuda(non_blocking=True)
        xn = xn.cuda(non_blocking=True)
        yn = yn.cuda(non_blocking=True)

        zs, xs, xn = model(xs, ys, xn, yn)

        ys = ys.t().contiguous()

        loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100)

        optimizer.step()

        err.update(loss.item())
        grd.update(grad_norm)

        progress.set_description('epoch %d %s %s' % (epoch + 1, err, grd))

    model.eval()

    err = AverageMeter('loss')
    cer = AverageMeter('cer')
    wer = AverageMeter('wer')
    ent = AverageMeter('ent')

    with torch.no_grad():
        progress = tqdm(test)
        for xs, ys, xn, yn in progress:

            xs = xs.cuda(non_blocking=True)
            ys = ys.cuda(non_blocking=True)
            xn = xn.cuda(non_blocking=True)
            yn = yn.cuda(non_blocking=True)

            zs, xs, xn = model(xs, ys, xn, yn)

            ys = ys.t().contiguous()

            loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

            xs = model.greedy_decode(xs)

            err.update(loss.item())
            ent.update(entropy(xs))

            hypothesis = decoder.unpad(xs, xn, labels)
            references = decoder.unpad(ys, yn, labels)

            for h, r in zip(hypothesis, references):
                cer.update(decoder.cer(h, r))
                wer.update(decoder.wer(h, r))

            progress.set_description('epoch %d %s %s %s %s' % (epoch + 1, err, cer, wer, ent))
        sys.stderr.write('\n')
