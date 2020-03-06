import math
import os

import librosa
import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset

FFT_SIZE = 1024
HOP_SIZE = 512
SPEC_DIMS = (513, 87)
DATA_MEAN = 0.1128162226528620
DATA_STD = 0.689340308041217

class SingingData(Dataset):
    def __init__(self, file_dir):
        self._files = mine_files(file_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        file_path = self._files[i]
        spec = np.load(file_path)

        trimmed = np.zeros(SPEC_DIMS)

        to = (min(SPEC_DIMS[0], spec.shape[0]), min(SPEC_DIMS[1], spec.shape[1]))
        trimmed[:to[0], :to[1]] = spec[:to[0], :to[1]]

        normalised = (trimmed - DATA_MEAN) / DATA_STD

        return from_numpy(normalised).view(1, SPEC_DIMS[0], SPEC_DIMS[1].float()


def make_spectrogram(audio):
    return np.abs(librosa.core.stft(audio, FFT_SIZE, HOP_SIZE))


def preprocess_data(audio_file):
    print("Processing %s" % (audio_file))
    x, sr = librosa.load(audio_file)

    chop_length = sr * 2

    chunks = []

    n_chunks = math.ceil(x.shape[0] / chop_length)
    for n in range(n_chunks):
        chunks.append(zero_pad(x[n * chop_length:(n + 1) * chop_length], chop_length))

    spectrograms = [ ]
    for chunk in chunks:
        spec = make_spectrogram(chunk)
        spectrograms.append(make_spectrogram(chunk))

    return spectrograms


def zero_pad(arr, to):
    x = arr.copy()
    if x.shape[0] < to:
        diff = to - x.shape[0]
        np.pad(x, (0, diff), 'constant', constant_values=0)
        return x
    else:
        return x[:to]


def mine_files(file_dir, ext='.npy'):
    files = []
    for entry in os.scandir(file_dir):
        if entry.is_dir():
            files += mine_files(entry.path)
            continue
        elif os.path.splitext(entry.name)[1] == ext:
            files.append(entry.path)

    return files
