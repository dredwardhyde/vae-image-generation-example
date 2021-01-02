import os
import sys
import time
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from colorama import Fore


# ================================================= MODEL ==============================================================
class VAE(nn.Module):
    def __init__(self, zsize):
        super(VAE, self).__init__()

        self.zsize = zsize

        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 2048, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(2048 * 4 * 4, zsize)
        self.fc2 = nn.Linear(2048 * 4 * 4, zsize)

        self.d1 = nn.Linear(zsize, 2048 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def encode(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = x.view(x.shape[0], 2048 * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], 2048, 4, 4)
        x = F.leaky_relu(x, 0.2)
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


# ============================================= LOSS FUNCTION ==========================================================
def loss_function(recon_x, x, mu, logvar):
    bce = torch.mean((recon_x - x) ** 2)
    kld = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return bce, kld * kl_weight


# ========================================== IMAGE PREPROCESSING =======================================================
def process_images(im_collection):
    data = [np.array(Image.fromarray(x).resize([im_size, im_size])).transpose((2, 0, 1)) for x in im_collection]
    x = np.asarray(data, dtype=np.float32) / 127.5 - 1.
    x = x.reshape(-1, 3, im_size, im_size)
    return x


# ============================================ HYPERPARAMETERS =========================================================
im_size = 128
batch_size = 256
z_size = 512
kl_weight = 1.7
train_epoch = 200
lr = 0.0008
gradient_clipping_value = 0.1
# ============================================= PREPARING DATASET ======================================================
im_collection = io.imread_collection('./cropped/*.png')
data_train = images = process_images(im_collection)
print("Train set size:", len(data_train))
batches_per_epoch = (len(data_train) // batch_size) + 1
print("Batches in 1 epoch: ", batches_per_epoch)
os.makedirs('results_reconstructed', exist_ok=True)
os.makedirs('results_generated', exist_ok=True)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
# ================================================ TRAINING MODEL ======================================================
vae = VAE(zsize=z_size)
vae.cuda()
vae.train()
vae.weight_init(mean=0, std=0.02)
vae_optimizer = optim.Adam(vae.parameters(), lr=lr)

for epoch in range(train_epoch):
    vae.train()
    reconstruction_loss = 0
    kullback_leibler_loss = 0
    epoch_start_time = time.time()
    i = 0
    training_pbar = tqdm(total=len(data_train),
                         position=0, leave=True,
                         file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    for _, x in enumerate(train_loader):
        # ============================================ TRAINING ========================================================
        vae.train()
        vae.zero_grad()
        x = x.cuda()
        rec, mu, logvar = vae(x)
        loss_re, loss_kl = loss_function(rec, x, mu, logvar)
        (loss_re + loss_kl).backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), gradient_clipping_value)
        vae_optimizer.step()
        reconstruction_loss += loss_re.item()
        kullback_leibler_loss += loss_kl.item()
        epoch_end_time = time.time() - epoch_start_time
        training_pbar.update(x.shape[0])
        # ============================================ VALIDATION ======================================================
        i += 1
        if i % batches_per_epoch == 0:
            training_pbar.close()
            reconstruction_loss /= batches_per_epoch
            kullback_leibler_loss /= batches_per_epoch
            print('\n[%d/%d] - epoch time: %.2f, reconstruction loss: %.9f, Kullback-Leibler loss: %.9f' % (
                (epoch + 1), train_epoch, epoch_end_time, reconstruction_loss, kullback_leibler_loss))
            reconstruction_loss = 0
            kullback_leibler_loss = 0
            with torch.no_grad():
                vae.eval()
                x_rec, _, _ = vae(x)
                result_sampled = torch.cat([x, x_rec]) * 0.5 + 0.5
                result_sampled = result_sampled.cpu()
                save_image(result_sampled.view(-1, 3, im_size, im_size),
                           'results_reconstructed/sample_' + str(epoch) + "_" + str(i) + '.png')
                sample = torch.randn(128, z_size).view(-1, z_size, 1, 1).cuda()
                x_rec = vae.decode(sample)
                result_sampled = x_rec * 0.5 + 0.5
                result_sampled = result_sampled.cpu()
                save_image(result_sampled.view(-1, 3, im_size, im_size),
                           'results_generated/sample_' + str(epoch) + '.png')
    torch.save(vae.state_dict(), "./weights_" + str(epoch) + ".pth")

# ============================================ TESTING =================================================================
vae.load_state_dict(torch.load("./weights_177.pth"))
os.makedirs('generated_results', exist_ok=True)
for i in range(0, 200):
    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1).cuda()
    x_rec = vae.decode(sample1)
    resultsample = x_rec * 0.5 + 0.5
    resultsample = resultsample.cpu()
    save_image(resultsample.view(-1, 3, im_size, im_size), 'generated_results/sample_' + str(i) + '.png')
