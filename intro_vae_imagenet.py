from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--train_path', type=str, default='/home/nutszebra/Downloads/m_ILSVRC/Data/CLS-LOC/train', metavar='N',
                    help='path for train images')
parser.add_argument('--test_path', type=str, default='/home/nutszebra/Downloads/m_ILSVRC/Data/CLS-LOC/val', metavar='N',
                    help='path for test images')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
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

kwargs = {'num_workers': 4,
          'pin_memory': True}
# load dataset
train_transform = transforms.Compose([transforms.RandomSizedCrop(256),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      ])
test_transform = transforms.Compose([transforms.Scale(256),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     ])
train_dataset = datasets.ImageFolder(args.train_path,
                                     transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
test_dataset = datasets.ImageFolder(args.test_path,
                                    transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          **kwargs)


class VAE(nn.Module):

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


def encode_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.Conv2d(input_features, output_features, 6, stride=2, padding=2),
                              nn.BatchNorm2d(output_features),
                              nn.LeakyReLU(0.2))
    return conv_unit


def decode_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, 4, stride=2, padding=1),
                              nn.BatchNorm2d(output_features),
                              nn.LeakyReLU(0.2))
    return conv_unit


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.nf = 64
        self.conv1 = encode_unit(3, self.nf)
        self.conv2 = encode_unit(self.nf, 2 * self.nf)
        self.conv3 = encode_unit(2 * self.nf, 4 * self.nf)
        self.conv4 = encode_unit(4 * self.nf, 8 * self.nf)
        self.conv5 = encode_unit(8 * self.nf, 8 * self.nf)
        self.fc1 = nn.Linear(64 * 8 * self.nf, 512)
        self.fc2 = nn.Linear(64 * 8 * self.nf, 512)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.nf = 64
        self.project = nn.Linear(512, 16 * 8 * self.nf, bias=False)
        self.batch_norm1d = nn.BatchNorm1d(16 * 8 * self.nf)
        self.dconv6 = decode_unit(8 * self.nf, 4 * self.nf)
        self.dconv5 = decode_unit(4 * self.nf, 2 * self.nf)
        self.dconv4 = decode_unit(2 * self.nf, 2 * self.nf)
        self.dconv3 = decode_unit(2 * self.nf, self.nf)
        self.dconv2 = decode_unit(self.nf, self.nf)
        self.dconv1 = decode_unit(self.nf, 3)

    def forward(self, z):
        h = self.batch_norm1d(self.project(z))
        h = F.leaky_relu(h, negative_slope=0.2, inplace=True)
        h = h.view(h.size(0), 8 * self.nf, 4, 4)
        h = self.dconv6(h)
        h = self.dconv5(h)
        h = self.dconv4(h)
        h = self.dconv3(h)
        h = self.dconv2(h)
        h = self.dconv1(h)
        return F.sigmoid(h)


def l_reg(mu, std):
    return - 0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2, dim=-1)


def loss_function(x, x_r,
                  z_mu, z_std,
                  z_r_mu, z_r_std,
                  z_pp_mu, z_pp_std,
                  z_r_detach_mu, z_r_detach_std,
                  z_pp_detach_mu, z_pp_detach_std):
        l_ae = torch.sum((x.reshape(-1, 3 * 256 * 256) - x_r.reshape(-1, 3 * 256 * 256)) ** 2, dim=-1)
        l_e_adv = l_reg(z_mu, z_std) + alpha * (F.relu(m - l_reg(z_r_detach_mu, z_r_detach_std)) + F.relu(m - l_reg(z_pp_detach_mu, z_pp_detach_std)))
        l_g_adv = alpha * (l_reg(z_r_mu, z_r_std) + l_reg(z_pp_mu, z_pp_std))
        loss = torch.mean(l_e_adv + l_g_adv + beta * l_ae)
        import IPython
        IPython.embed()
        return loss


alpha = 0.5
beta = 0.5
m = 1.0

encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0002)


def train(epoch):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        x = data
        optimizer.zero_grad()

        z_mu, z_logvar = encoder(x)
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        z = eps.mul(z_std).add_(z_mu)

        x_r = decoder(z)
        z_r_mu, z_r_logvar = encoder(x_r)
        z_r_std = torch.exp(0.5 * z_r_logvar)
        eps = torch.randn_like(z_r_std)
        z_r = eps.mul(z_r_std).add_(z_r_mu)

        z_r_detach_mu, z_r_detach_logvar = encoder(x_r.detach())
        z_r_detach_std = torch.exp(0.5 * z_r_detach_logvar)
        eps = torch.randn_like(z_r_detach_std)
        z_r_detach = eps.mul(z_r_detach_std).add_(z_r_detach_mu)

        z_p = torch.randn_like(z)
        x_p = decoder(z_p)

        z_pp_mu, z_pp_logvar = encoder(x_p)
        z_pp_std = torch.exp(0.5 * z_pp_logvar)
        eps = torch.randn_like(z_pp_std)
        z_pp = eps.mul(z_pp_std).add_(z_pp_mu)

        z_pp_detach_mu, z_pp_detach_logvar = encoder(x_p.detach())
        z_pp_detach_std = torch.exp(0.5 * z_pp_detach_logvar)
        eps = torch.randn_like(z_pp_detach_std)
        z_pp_detach = eps.mul(z_pp_detach_std).add_(z_pp_detach_mu)

        loss = loss_function(x, x_r,
                             z_mu, z_std,
                             z_r_mu, z_r_std,
                             z_pp_mu, z_pp_std,
                             z_r_detach_mu, z_r_detach_std,
                             z_pp_detach_mu, z_pp_detach_std)

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


def test(epoch):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            x = data

            z_mu, z_logvar = encoder(x)
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_std)
            z = eps.mul(z_std).add_(z_mu)

            x_r = decoder(z)
            z_r_mu, z_r_logvar = encoder(x_r)
            z_r_std = torch.exp(0.5 * z_r_logvar)
            eps = torch.randn_like(z_r_std)
            z_r = eps.mul(z_r_std).add_(z_r_mu)

            z_r_detach_mu, z_r_detach_logvar = encoder(x_r.detach())
            z_r_detach_std = torch.exp(0.5 * z_r_detach_logvar)
            eps = torch.randn_like(z_r_detach_std)
            z_r_detach = eps.mul(z_r_detach_std).add_(z_r_detach_mu)

            z_p = torch.randn_like(z)
            x_p = decoder(z_p)

            z_pp_mu, z_pp_logvar = encoder(x_p)
            z_pp_std = torch.exp(0.5 * z_pp_logvar)
            eps = torch.randn_like(z_pp_std)
            z_pp = eps.mul(z_pp_std).add_(z_pp_mu)

            z_pp_detach_mu, z_pp_detach_logvar = encoder(x_p.detach())
            z_pp_detach_std = torch.exp(0.5 * z_pp_detach_logvar)
            eps = torch.randn_like(z_pp_detach_std)
            z_pp_detach = eps.mul(z_pp_detach_std).add_(z_pp_detach_mu)

            loss = loss_function(x, x_r,
                                 z_mu, z_std,
                                 z_r_mu, z_r_std,
                                 z_pp_mu, z_pp_std,
                                 z_r_detach_mu, z_r_detach_std,
                                 z_pp_detach_mu, z_pp_detach_std)

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 512).to(device)
        sample = decoder(sample).cpu()
        sample = sample.view(64, 3, 256, 256)
        save_image(sample,
                   'results/sample_' + str(epoch) + '.png')
