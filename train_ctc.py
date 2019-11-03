import sys
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import StepLR

import numpy as np

from data import Labels, AudioDataset, DataLoader, collate_fn, collate_fn_ctc, BucketingSampler

from tqdm import tqdm

from model import AcousticModel
from utils import AverageMeter

from torch_baidu_ctc import ctc_loss
from pytorch_edit_distance import remove_repetitions, remove_blank, AverageWER, AverageCER


torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

labels = Labels()

model = AcousticModel(40, 512, 256, len(labels), n_layers=3, dropout=0.3)

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

train.filter_by_conv(model.conv)
train.filter_by_length(400)

test.filter_by_conv(model.conv)
test.filter_by_length(500)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=500, gamma=0.99)

model.cuda()

sampler = BucketingSampler(train, 32)

train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_ctc, batch_sampler=sampler)
test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn, batch_size=32)

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

for epoch in range(10):

    sampler.shuffle(epoch)

    model.train()

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')

    progress = tqdm(train)
    for xs, ys, xn, yn in progress:

        optimizer.zero_grad()

        xs, xn = model(xs, xn)
        xs = log_softmax(xs, dim=-1)

        loss = ctc_loss(xs, ys, xn, yn, average_frames=False, reduction="mean")
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100)

        optimizer.step()
        scheduler.step()

        err.update(loss.item())
        grd.update(grad_norm)

        lr = scheduler.get_lr()[0]

        progress.set_description('epoch %d %.6f %s %s' % (epoch + 1, lr, err, grd))

    model.eval()

    err = AverageMeter('loss')
    cer = AverageCER(blank, space)
    wer = AverageWER(blank, space)

    with torch.no_grad():
        progress = tqdm(test)
        for xs, ys, xn, yn in progress:

            xs, xn = model(xs.cuda(non_blocking=True), xn)
            xs = log_softmax(xs, dim=-1)

            loss = ctc_loss(xs, torch.cat(ys), xn, yn, average_frames=False, reduction="mean")

            err.update(loss.item())

            xs = xs.argmax(2).t().type(torch.int)
            ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True).cuda(non_blocking=True)
            xn = xn.cuda(non_blocking=True)
            yn = yn.cuda(non_blocking=True)

            remove_repetitions(xs, xn)
            remove_blank(xs, xn, blank)

            cer.update(xs, ys, xn, yn)
            wer.update(xs, ys, xn, yn)

            progress.set_description('epoch %d %s %s %s' % (epoch + 1, err, cer, wer))

    sys.stderr.write('\n')

    torch.save(model.state_dict(), 'exp/am.bin')
