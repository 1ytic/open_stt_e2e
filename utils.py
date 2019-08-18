import numpy as np
import scipy.stats


def entropy(labels):
    value, counts = np.unique(labels.flatten(), return_counts=True)
    return scipy.stats.entropy(counts)


class AverageMeter(object):

    def __init__(self, title=''):
        self.title = title
        self.count = 0
        self.sum = 0
        self.sum2 = 0
        self.avg = 0
        self.std = 0

    def update(self, x):
        self.count += 1
        self.sum += x
        self.sum2 += x * x
        self.avg = self.sum / self.count
        self.std = np.sqrt(self.sum2 / self.count - self.avg * self.avg)

    def __str__(self):
        return '%s %.4f\u00B1%.2f' % (self.title, self.avg, self.std)
