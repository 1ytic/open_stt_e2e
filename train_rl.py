import sys
import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu, elu
from torch.optim.lr_scheduler import StepLR

from data import Labels, AudioDataset, DataLoaderCuda, collate_audio, BucketingSampler

from model import Transducer
from utils import AverageMeter

from warp_rnnt import rnnt_loss
import pytorch_edit_distance


torch.backends.cudnn.benchmark = True
torch.manual_seed(2)
np.random.seed(2)

labels = Labels()

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.4)
model.load_state_dict(torch.load('exp/asr.bin', map_location='cpu'))
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

train.filter_by_conv(model.encoder.conv)
train.filter_by_length(400)

test.filter_by_conv(model.encoder.conv)
test.filter_by_length(500)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=250, gamma=0.99)

sampler = BucketingSampler(train, 32)

train = DataLoaderCuda(train, collate_fn=collate_audio, batch_sampler=sampler)
test = DataLoaderCuda(test, collate_fn=collate_audio, batch_size=16)

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

N = 10
alpha = 0.01

for epoch in range(20):

    sampler.shuffle(epoch)

    err = AverageMeter('loss')
    grd = AverageMeter('gradient')
    rwd = AverageMeter('reward')

    num_batch = 0

    for xs, ys, xn, yn in train:

        optimizer.zero_grad()

        model.train()

        zs, xs, xn = model(xs, ys.t(), xn, yn)

        model.eval()

        with torch.no_grad():

            xs_e = xs.repeat(N, 1, 1)
            xn_e = xn.repeat(N)
            ys_e = ys.repeat(N, 1)
            yn_e = yn.repeat(N)

            hs_e = model.greedy_decode(xs_e, sampled=True)

            pytorch_edit_distance.remove_blank(hs_e, xn_e, blank)

            Err = pytorch_edit_distance.wer(hs_e, ys_e, xn_e, yn_e, blank, space)

            xn_e_safe = torch.max(xn_e, torch.ones_like(xn_e)).float()

            SymAcc = 1 - 0.5 * Err * (1 + yn_e.float() / xn_e_safe)

            rewards = relu(SymAcc).reshape(N, -1)

            hs_e = hs_e.reshape(N, len(xs), -1)
            xn_e = xn_e.reshape(N, len(xs))

        model.train()

        rewards = rewards.cuda()

        rwd.update(rewards.mean().item())

        rewards -= rewards.mean(dim=0)

        elu(rewards, alpha=0.5, inplace=True)

        total_loss = 0

        if alpha > 0:

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = alpha * nll.mean()
            loss.backward(retain_graph=True)

            total_loss = loss.item()

        for n in range(N):

            ys = hs_e[n]
            yn = xn_e[n]

            ys = ys[:, :yn.max()].contiguous()

            zs = model.forward_decoder(xs, ys.t(), yn)

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = nll * rewards[n]

            loss = loss.mean() / N

            loss.backward(retain_graph=True)

            total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        scheduler.step()

        err.update(total_loss)
        grd.update(grad_norm)
        
        train.set_description('epoch %d %s %s %s' % (epoch + 1, err, grd, rwd))

        num_batch += 1
        if num_batch == 500:
            train.close()
            break

    model.eval()

    err = AverageMeter('loss')
    cer = pytorch_edit_distance.AverageCER(blank, space)
    wer = pytorch_edit_distance.AverageWER(blank, space)

    with torch.no_grad():

        for xs, ys, xn, yn in test:

            zs, xs, xn = model(xs, ys.t(), xn, yn)

            loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

            xs = model.greedy_decode(xs)

            err.update(loss.item())

            pytorch_edit_distance.remove_blank(xs, xn, blank)

            wer.update(xs, ys, xn, yn)
            cer.update(xs, ys, xn, yn)

            test.set_description('epoch %d %s %s %s' % (epoch + 1, err, cer, wer))

    sys.stderr.write('\n')

    torch.save(model.state_dict(), 'exp/rl.bin')
