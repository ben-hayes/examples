from __future__ import print_function
import argparse
import librosa
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import random_split
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import dataset


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

train_split = 0.9
data = dataset.SingingData('data')
train_len = int(len(data) * train_split)
train_data, test_data = random_split(data, (train_len, len(data) - train_len))

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv21 = nn.Conv2d(2, 4, 5, padding=2)
        self.conv22 = nn.Conv2d(2, 4, 5, padding=2)

        self.conv3 = nn.Conv2d(4, 2, 5, padding=2)
        self.conv4 = nn.Conv2d(2, 1, 5, padding=2)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        return self.conv21(h1), self.conv22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.conv3(z))
        return torch.sigmoid(self.conv4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def save_audio(tensor, file_out, n_row=n):
    y = tensor.numpy() * DATA_RANGE
    x = librosa.core.istft(x, dataset.HOP_SIZE, dataset.FFT_SIZE)
    librosa.output.write_wav(file_out, x, 22050)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                for n, item in enumerate(recon_batch):
                    if n < 10:
                        x = item.view(513, 87)
                        save_audio(x, 'recon_%d_%d.wav' % (i, n))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(5, 4, 513, 87).to(device)
            sample = model.decode(sample).cpu()
            for n, item in enumerate(sample):
                    if n < 10:
                        x = item.view(513, 87)
                        save_audio(x, 'sample_%d_%d.wav' % (i, n))
