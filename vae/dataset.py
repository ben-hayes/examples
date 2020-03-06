import math
import os

import librosa
from torch.utils.data import Dataset

FFT_SIZE = 1024
HOP_SIZE = 512

class SingingData(Dataset):
    def __init__(self, file_dir):
        self._files = self._mine_files(file_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        file_path = self._files[i]
        x, sr = librosa.load(file_path)
        spec = 0


def make_spectrogram(audio):
    return np.abs(librosa.core.stft(audio, FFT_SIZE, HOP_SIZE))


def preprocess_data(audio_file):
    x, sr = librosa.load(audio_file)

    chop_length = sr * 2

    chunks = []

    n_chunks = math.ceil(x.shape[0] / chop_length)
    for n in n_chunks:
        chunks.append(zero_pad(x[n * chop_length:(n + 1) * chop_length], chop_length))

    spectrograms = [ ]
    for chunk in chunks:
        spectrograms.append(make_spectrogram(audio))


def zero_pad(arr, to):
    x = arr.copy()
    if x.shape[0] < chop_length:
        diff = x.shape[0] - chop_length
        np.pad(x, (0, diff), 'constant', contant_values=0)
        return x
    else:
        return x[:to]


def mine_files(file_dir):
    files = []
    for entry in os.scandir(file_dir):
        if entry.is_dir():
            files += mine_files(entry.path)
            continue
        elif os.path.splitext(entry.name)[1] == ".wav":
            files.append(entry.path)

    return files
