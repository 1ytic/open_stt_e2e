import sys
import torch
import numpy as np
import torch.nn as nn
import torch_edit_distance
from torch.nn.functional import relu, elu
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_edit_distance import compute_wer, remove_blank, AverageWER, AverageCER
from data import Labels, split_train_dev_test
from model import Transducer
from utils import AverageMeter
from warp_rnnt import rnnt_loss

torch.backends.cudnn.benchmark = True
torch.manual_seed(2)
np.random.seed(2)

labels = Labels()

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

model_path = 'runs/rnnt_bs32x4_gn200_beta0.5/model10.bin'

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.3)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.cuda()

train, dev, test = split_train_dev_test(
    '/media/lytic/STORE/ru_open_stt_wav',
    labels, model.am.conv, batch_size=16
)

parameters = [
    {'params': model.fc.parameters(), 'lr': 3e-6},
    {'params': model.am.parameters(), 'lr': 3e-6},
    {'params': model.lm.parameters(), 'lr': 3e-6}
]

optimizer = torch.optim.Adam(parameters, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

K = 10
alpha = 0.01
beta = 0.01
gamma = 0.5

step = 0
writer = SummaryWriter(comment='_rl_bs16x4_beta0.01_lr3e-6')

model.eval()

with torch.no_grad():

    temperature = 3
    prediction = []
    prior = 0

    for xs, ys, xn, yn in dev:

        xs, xn = model.forward_acoustic(xs, xn)

        xs = model.greedy_decode(xs, argmax=False)

        xs = xs.exp().view(-1, len(labels))

        prediction.append(xs.argmax(1).cpu())
        prior += xs.sum(dim=0)

        dev.set_description('Prior %.5f' % (prior.std().item()))

    prediction = torch.cat(prediction)
    prior = (prior / prediction.size(0)).log() / temperature

for epoch in range(1, 11):

    train.shuffle(epoch)

    err = AverageMeter('Loss/train')
    ent = AverageMeter('Entropy/train')
    grd = AverageMeter('Gradient/train')
    rwd = AverageMeter('Reward/train')

    optimizer.zero_grad()

    for xs, ys, xn, yn in train:

        step += 1

        model.eval()

        with torch.no_grad():

            hs, hn = model.forward_acoustic(xs, xn)

            hs_k = hs.repeat(K, 1, 1)
            hn_k = hn.repeat(K)
            ys_k = ys.repeat(K, 1)
            yn_k = yn.repeat(K)

            hs_k = model.greedy_decode(hs_k, prior, sampled=True)

            remove_blank(hs_k, hn_k, blank)

            WER = compute_wer(hs_k, ys_k, hn_k, yn_k, blank, space)

            SymAcc = 1 - 0.5 * WER * (1 + yn_k.float() / hn_k.clamp_min(1).float())

            rewards = relu(SymAcc).reshape(K, -1).cuda()

            rewards_mean = rewards.mean().item()

            rewards -= rewards.mean(dim=0)

            elu(rewards, alpha=gamma, inplace=True)

            hs_k = hs_k.reshape(K, len(xs), -1)
            hn_k = hn_k.reshape(K, len(xs))

        model.train()

        zs, xs, xn = model(xs, ys.t(), xn, yn)

        loss1 = rnnt_loss(zs, ys, xn, yn).mean()

        loss2 = -(zs.exp() * zs).sum(dim=-1).mean()

        for k in range(K):

            ys = hs_k[k]
            yn = hn_k[k]

            ys = ys[:, :yn.max()].contiguous()

            zs = model.forward_language(ys.t(), yn)

            zs = model.forward_joint(xs, zs)

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = nll * rewards[k]

            loss = loss.mean() / K

            loss.backward(retain_graph=True)

        loss = 0

        if alpha > 0:
            loss += alpha * loss1

        if beta > 0:
            loss -= beta * loss2

        loss.backward()

        err.update(loss1.item())
        ent.update(loss2.item())
        rwd.update(rewards_mean)

        writer.add_scalar(err.title + '/steps', loss1.item(), step)
        writer.add_scalar(ent.title + '/steps', loss2.item(), step)
        writer.add_scalar(rwd.title + '/steps', rewards_mean, step)

        if step % 4 > 0:
            continue

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        grd.update(grad_norm)

        writer.add_scalar(grd.title + '/steps', grad_norm, step)
        
        train.set_description('Epoch %d %s %s %s %s' % (epoch, err, ent, grd, rwd))

        if step % 200 == 0:
            train.close()
            break

    model.eval()

    for i, lr in enumerate(scheduler.get_lr()):
        writer.add_scalar('LR/%d' % i, lr, epoch)

    err.summary(writer, epoch)
    ent.summary(writer, epoch)
    grd.summary(writer, epoch)
    rwd.summary(writer, epoch)

    err = AverageMeter('Loss/test')
    ent = AverageMeter('Entropy/test')
    cer = AverageCER(blank, space)
    wer = AverageWER(blank, space)

    with torch.no_grad():

        for xs, ys, xn, yn in test:

            zs, xs, xn = model(xs, ys.t(), xn, yn)

            loss1 = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

            loss2 = -(zs.exp() * zs).sum(dim=-1).mean()

            xs = model.greedy_decode(xs, prior)

            err.update(loss1.item())
            ent.update(loss2.item())

            remove_blank(xs, xn, blank)

            cer.update(xs, ys, xn, yn)
            wer.update(xs, ys, xn, yn)

            test.set_description('Epoch %d %s %s %s %s' % (epoch, err, ent, cer, wer))

    sys.stderr.write('\n')

    err.summary(writer, epoch)
    ent.summary(writer, epoch)
    cer.summary(writer, epoch)
    wer.summary(writer, epoch)

    writer.flush()

    # torch.save(model.state_dict(), writer.log_dir + '/model%d.bin' % epoch)

writer.close()
