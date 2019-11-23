import argparse
import librosa
import numpy as np
from tqdm import tqdm
from os.path import join, isfile
from joblib import Parallel, delayed
from python_speech_features import get_filterbanks, sigproc

samplerate = 16000
nfft = 512
winlen = 0.025 * samplerate
winstep = 0.01 * samplerate
banks = get_filterbanks(40, nfft, samplerate).transpose()


def job(input_name, output_name):
    audio, _ = librosa.load(input_name, mono=True, sr=samplerate)
    if len(audio) == 0:
        return False
    signal = sigproc.preemphasis(audio, 0.97)
    x = sigproc.framesig(signal, winlen, winstep, np.hanning)
    if len(x) == 0:
        return False
    x = sigproc.powspec(x, nfft)
    x = np.dot(x, banks)
    x = np.where(x == 0, np.finfo(float).eps, x)
    x = np.log(x).astype(dtype=np.float32)
    if np.isnan(np.sum(x)):
        return False
    np.save(output_name, x)
    return True


parser = argparse.ArgumentParser(description='Compute features')
parser.add_argument('--manifest', type=str)
parser.add_argument('--jobs', type=int, default=8)
args = parser.parse_args()

prefix = args.manifest.replace('.csv', '')

print(prefix)

files = dict()

with open(args.manifest) as f:

    for line in tqdm(f.readlines()):

        path = line.split(',')[0]

        audio_path = join(prefix, path)
        if not isfile(audio_path):
            continue

        numpy_path = join(prefix, path.replace('.wav', '.npy'))
        if isfile(numpy_path):
            continue

        files[audio_path] = numpy_path

tasks = []
for audio_path, numpy_path in files.items():
    tasks.append(delayed(job)(audio_path, numpy_path))

print('Tasks:', len(tasks))

results = Parallel(n_jobs=args.jobs, backend='multiprocessing', verbose=1)(tasks)

print('Success:', sum(results))
