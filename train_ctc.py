import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_edit_distance import collapse_repeated, remove_blank, AverageWER, AverageCER

from data import Labels, split_train_dev_test
from model import AcousticModel
from utils import AverageMeter

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

labels = Labels()

blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()

model = AcousticModel(40, 512, 256, len(labels), n_layers=3, dropout=0.3)
model.cuda()

train, dev, test = split_train_dev_test(
    '/media/lytic/STORE/ru_open_stt_wav',
    labels, model.conv, batch_size=32
)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

ctc_loss = nn.CTCLoss(blank=labels.blank(), reduction='none', zero_infinity=True)

step = 0
writer = SummaryWriter(comment="_ctc_bs32x4_gn200")

for epoch in range(1, 21):

    train.shuffle(epoch)

    model.train()

    err = AverageMeter('Loss/train')
    ent = AverageMeter('Entropy/train')
    grd = AverageMeter('Gradient/train')

    optimizer.zero_grad()

    for xs, ys, xn, yn in train:

        step += 1

        xs, xn = model(xs, xn)

        loss1 = ctc_loss(xs, ys, xn, yn).mean()

        loss2 = -(xs.exp() * xs).sum(dim=-1).mean()

        loss1.backward()

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

            xs, xn = model(xs, xn)

            xs = xs.exp().view(-1, len(labels))

            prediction.append(xs.argmax(1).cpu())
            prior += xs.sum(dim=0)

            dev.set_description('Epoch %d Prior %.5f' % (epoch, prior.std().item()))

        prediction = torch.cat(prediction)
        prior = (prior / prediction.size(0)).log() / temperature

        writer.add_histogram('Prediction', prediction[prediction != labels.blank()], epoch)
        writer.add_histogram('Prior', prior, epoch)

        for xs, ys, xn, yn in test:

            xs, xn = model(xs, xn)

            loss1 = ctc_loss(xs, ys, xn, yn).mean()

            loss2 = -(xs.exp() * xs).sum(dim=-1).mean()

            err.update(loss1.item())
            ent.update(loss2.item())

            xs = xs - prior
            xs = xs.argmax(2).t().type(torch.int)

            collapse_repeated(xs, xn)
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
