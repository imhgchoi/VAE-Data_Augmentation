import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class VanillaVAE(nn.Module):
    def __init__(self, config):
        super(VanillaVAE, self).__init__()
        self.config = config
        self.device = config.device
        self.porify_p = config.vae_porify
        self.logvar_factor = config.vae_logvar_factor
        self.logvar_clip = config.vae_logvar_clip
        self.start_std = config.vae_start_std
        self.start_std = False

        encoder_dim = [config.window] + config.vae_enc_dim
        decoder_dim = [config.vae_repr_dim] + config.vae_dec_dim

        self.encoder = nn.Sequential()
        for i in range(len(encoder_dim)-1):
            self.add_enc_layer(encoder_dim[i], encoder_dim[i+1], i+1)

        self.mu = nn.Linear(encoder_dim[-1], config.vae_repr_dim)
        self.logvar = nn.Linear(encoder_dim[-1], config.vae_repr_dim)

        self.decoder = nn.Sequential()
        for i in range(len(decoder_dim)-1) :
            self.add_dec_layer(decoder_dim[i], decoder_dim[i+1], i+1)

        self.out = nn.Linear(decoder_dim[-1], config.window)

        self.criterion = self.ELBO
        self.to(config.device)

    def add_enc_layer(self, indim, outdim, i):
        self.encoder.add_module("fc{}".format(i), nn.Linear(indim, outdim))
        self.encoder.add_module("elu{}".format(i), nn.ELU())
        # self.encoder.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def add_dec_layer(self, indim, outdim, i):
        self.decoder.add_module("fc{}".format(i), nn.Linear(indim, outdim))
        self.decoder.add_module("elu{}".format(i), nn.ELU())
        # self.decoder.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def porify(self, X):
        mask = torch.FloatTensor(X.shape).uniform_() > self.porify_p
        mask = mask.to(self.device)
        return X * mask

    def reparameterize(self, mu, logvar, step):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) / self.logvar_factor
        return mu + eps * std * int(self.start_std)

    def forward(self, x, step):
        x = self.porify(x)
        for layer in self.encoder:
            x = layer(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)
        x = self.reparameterize(mu, logvar, step)

        for layer in self.decoder:
            x = layer(x)
        x = self.out(x)
        return x, mu, logvar

    def ELBO(self, out, actual, mu, logvar, step):
        loss = nn.MSELoss(reduction='mean')
        MSE = loss(out, actual)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD, MSE, KLD

    def augment(self, x, augNum):
        self.eval()
        for layer in self.encoder:
            x = layer(x)
        mu = self.mu(x)
        logvar = self.logvar(x) / self.logvar_factor
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)

        augmented = []
        for _ in range(augNum):
            x = self.reparameterize(mu, logvar, 99999)
            for layer in self.decoder:
                x = layer(x)
            out = self.out(x)
            augmented.append(out)

        return augmented