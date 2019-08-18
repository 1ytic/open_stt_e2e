import editdistance as ed


def uncat(ys, yn, labels):
    offset = 0
    sequences = []
    for n in yn:
        s = ys[offset:offset + n]
        s = labels.string(s)
        sequences.append(s)
        offset += n
    return sequences


def unpad(ys, yn, labels, remove_repetitions=False):
    sequences = []
    for i, n in enumerate(yn):
        s = labels.string(ys[i, :n], remove_repetitions)
        sequences.append(s)
    return sequences


def wer(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    return ed.eval(s1, s2) / len(s2)


def cer(s1, s2):
    s1 = s1.replace(' ', '')
    s2 = s2.replace(' ', '')
    return ed.eval(s1, s2) / len(s2)
