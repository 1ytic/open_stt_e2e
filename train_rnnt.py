import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_edit_distance import remove_blank, AverageWER, AverageCER
from data import Labels, split_train_dev_test
from model import Transducer
from utils import AverageMeter
from warp_rnnt import rnnt_loss

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
np.random.seed(1)

labels = Labels()

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.3,
                   am_checkpoint='runs/ctc_bs32x4_gn200/model19.bin',
                   lm_checkpoint='runs/lm_bptt8_bs64_gn1_do0.3/model10.bin')
model.cuda()

train, dev, test = split_train_dev_test(
    '/media/lytic/STORE/ru_open_stt_wav',
    labels, model.am.conv, batch_size=32
)

parameters = [
    {'params': model.fc.parameters(), 'lr': 3e-5},
    {'params': model.am.parameters(), 'lr': 3e-5},
    {'params': model.lm.parameters(), 'lr': 3e-5}
]

optimizer = torch.optim.Adam(parameters, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.99)

beta = 0.5

step = 0
writer = SummaryWriter(comment='_rnnt_bs32x4_gn200_beta0.5')

for epoch in range(1, 11):

    train.shuffle(epoch)

    model.train()

    err = AverageMeter('Loss/train')
    ent = AverageMeter('Entropy/train')
    grd = AverageMeter('Gradient/train')

    optimizer.zero_grad()

    for xs, ys, xn, yn in train:

        step += 1

        zs, xs, xn = model(xs, ys.t(), xn, yn)

        loss1 = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

        loss2 = -(zs.exp() * zs).sum(dim=-1).mean()

        loss = loss1 - beta * loss2

        loss.backward()

        err.update(loss1.item())
        ent.update(loss2.item())

        writer.add_scalar(err.title + '/steps', loss1.item(), step)
        writer.add_scalar(ent.title + '/steps', loss2.item(), step)

        if step % 4 > 0:
            continue

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        grd.update(grad_norm)

        writer.add_scalar(grd.title + '/steps', grad_norm, step)

        train.set_description('Epoch %d %s %s %s' % (epoch, err, ent, grd))

    model.eval()

    for i, lr in enumerate(scheduler.get_lr()):
        writer.add_scalar('LR/%d' % i, lr, epoch)

    err.summary(writer, epoch)
    ent.summary(writer, epoch)
    grd.summary(writer, epoch)

    err = AverageMeter('Loss/test')
    ent = AverageMeter('Entropy/test')
    cer = AverageCER(blank, space)
    wer = AverageWER(blank, space)

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

            dev.set_description('Epoch %d Prior %.5f' % (epoch, prior.std().item()))

        prediction = torch.cat(prediction)
        prior = (prior / prediction.size(0)).log() / temperature

        writer.add_histogram('Prediction', prediction[prediction != labels.blank()], epoch)
        writer.add_histogram('Prior', prior, epoch)

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

    torch.save(model.state_dict(), writer.log_dir + '/model%d.bin' % epoch)

writer.close()
