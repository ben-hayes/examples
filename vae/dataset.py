import os

import librosa
from torch.utils.data import Dataset

class SingingData(Dataset):
    def __init__(self, file_dir):
        self._files = self._mine_files(file_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        file_path = self._files[i]
        x, sr = librosa.load(file_path)
        spec = 0

    def _mine_files(self, file_dir):
        files = []
        for entry in os.scandir(file_dir):
            if entry.is_dir():
                files += self._mine_files(entry.path)
                continue
            elif os.path.splitext(entry.name)[1] == ".wav":
                files.append(entry.path)

        return files

