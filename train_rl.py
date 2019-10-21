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

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.4)

model.load_state_dict(torch.load('exp/open-stt-rnnt/asr.bin', map_location='cpu'))

train = AudioDataset('/media/lytic/STORE/ru_open_stt_wav/public_youtube1120_hq.txt', labels)
test = AudioDataset('/media/lytic/STORE/ru_open_stt_wav/public_youtube700_val.txt', labels)

train.filter_by_conv(model.encoder.conv)
train.filter_by_length(400)

test.filter_by_conv(model.encoder.conv)
test.filter_by_length(200)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=300, gamma=0.99)

model.cuda()

sampler = BucketingSampler(train, 32)

train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_sampler=sampler)
test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_size=16)

N = 10
alpha = 0.01

for epoch in range(5):

    sampler.shuffle(epoch)

    model.train()

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')
    rwd = AverageMeter('reward')

    num_batch = 0

    progress = tqdm(train)
    for xs, ys, xn, yn in progress:

        optimizer.zero_grad()

        xs = xs.cuda(non_blocking=True)
        ys = ys.cuda(non_blocking=True)
        xn = xn.cuda(non_blocking=True)
        yn = yn.cuda(non_blocking=True)

        zs, xs, xn = model(xs, ys, xn, yn)

        actions = []
        lengths = []
        rewards = torch.empty((N, len(xn)), dtype=torch.float32)

        with torch.no_grad():

            ys = ys.t().contiguous()

            references = decoder.unpad(ys, yn, labels)

            for n in range(N):

                hypothesis = model.greedy_decode(xs, sampled=True)
                hypothesis = decoder.unpad(hypothesis, xn, labels)

                temp1 = []
                temp2 = []

                for i, (h, r) in enumerate(zip(hypothesis, references)):
                    codes = labels(h)
                    temp1.append(torch.tensor(codes, dtype=torch.int))
                    temp2.append(len(codes))
                    
                    # Reward shaping
                    # http://www.apsipa.org/proceedings/2018/pdfs/0001934.pdf
                    
                    N_ref = len(r)
                    N_hyp = max(len(codes), 1)
                    Err = wer(h, r)
                    SymAcc = 1 - Err / 2 * (1 + N_ref / N_hyp)
                    
                    rewards[n, i] = max(SymAcc, 0)

                temp1 = torch.nn.utils.rnn.pad_sequence(temp1)
                temp2 = torch.tensor(temp2, dtype=torch.int)

                actions.append(temp1)
                lengths.append(temp2)

        rewards = rewards.cuda()

        rwd.update(rewards.mean().item())

        rewards -= rewards.mean(dim=0)

        total_loss = 0

        if alpha > 0:

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = alpha * nll.mean()
            loss.backward(retain_graph=True)

            total_loss = loss.item()

        for n in range(N):

            ys = actions[n].cuda()
            yn = lengths[n].cuda()

            zs = model.forward_decoder(xs, ys, yn)

            nll = rnnt_loss(zs, ys.t().contiguous(), xn, yn)

            loss = nll * rewards[n]

            loss = loss.mean() / N

            loss.backward(retain_graph=True)

            total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        scheduler.step()

        err.update(total_loss)
        grd.update(grad_norm)
        
        progress.set_description('epoch %d %s %s %s' % (epoch + 1, err, grd, rwd))

        num_batch += 1
        if num_batch == 500:
            progress.close()
            break

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
