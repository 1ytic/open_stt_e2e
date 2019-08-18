import argparse
import numpy as np
from tqdm import tqdm
from os.path import join, isfile
from data import Labels
from joblib import Parallel, delayed

labels = Labels()


def job(text_path, numpy_path):

    with open(text_path, 'r', encoding='utf8') as file:
        text = file.read()

    if not labels.is_accepted(text):
        return None

    required_frames = labels.required_frames(text)
    actual_frames = len(np.load(numpy_path))

    if required_frames > actual_frames:
        return None

    return '%s,%d,%s' % (numpy_path, actual_frames, text)


parser = argparse.ArgumentParser(description='Collect utterances')
parser.add_argument('--manifest', type=str)
parser.add_argument('--jobs', type=int, default=8)
args = parser.parse_args()

prefix = args.manifest.replace('.csv', '')

print(prefix)

tasks = []

with open(args.manifest) as f:

    progress = tqdm(f.readlines())

    for line in progress:

        path = line.split(',')[0]

        text_path = join(prefix, path.replace('.wav', '.txt'))
        if not isfile(text_path):
            continue

        numpy_path = join(prefix, path.replace('.wav', '.npy'))
        if not isfile(numpy_path):
            continue

        tasks.append(delayed(job)(text_path, numpy_path))

print('Tasks:', len(tasks))

results = Parallel(n_jobs=args.jobs, backend='multiprocessing', verbose=1)(tasks)

utterances = sorted([r for r in results if r is not None])

print('Success:', len(utterances))

with open(prefix + '.txt', 'w', encoding='utf8') as file:
    file.write('path,frames,text\n')
    file.writelines(utterances)
