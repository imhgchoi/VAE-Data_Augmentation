import torch
import torch.nn as nn
import torch.optim as optim
import dill
import numpy as np
import matplotlib.pyplot as plt
import pdb

class Trainer():
    def __init__(self, config, datasets):
        self.config = config
        self.device = config.device
        self.lr = config.lr

        self.tr_data = datasets[0]
        self.vl_data = datasets[1]
        self.te_data = datasets[1]

    def train(self, model):
        if self.config.model in ['ae'] :
            self.train_ae(model)
        elif self.config.model in ['vae'] :
            self.train_vae(model)

    def train_vae(self, model):
        if self.config.reset_model :
            tr_losses, vl_losses = [], []
        else :
            model.load_state_dict(torch.load('./out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name)))
            with open('./out/{0}/loss/{0}_perf_idx_{1}.pkl'.format(self.config.model, self.config.model_name), 'rb') as f :
                tr_losses, vl_losses = dill.load(f)
            f.close()

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = model.criterion
        step = 0
        for epoch in range(1, self.config.epoch_num+1) :
            for tr_X, _ in self.tr_data.dataloader :
                step +=1

                # TRAIN
                model.train()
                tr_recon, mu, logvar = model(tr_X, step)
                tr_loss, tr_mse, tr_kld = criterion(tr_recon, tr_X,  mu, logvar, step)
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()
                if tr_mse.item() < 0.8 :
                    model.start_std = True

                # VALIDATE
                model.eval()
                tmp_losses, tmp_mses, tmp_klds = [], [], []
                for vl_X, _ in self.vl_data.dataloader :
                    vl_recon, mu, logvar = model(vl_X, step)
                    vl_loss, vl_mse, vl_mse = criterion(tr_recon, tr_X,  mu, logvar, step)
                    tmp_losses.append(vl_loss.item())
                    tmp_mses.append(vl_mse.item())
                    tmp_klds.append(vl_mse.item())
                vl_loss = sum(tmp_losses) / len(tmp_losses)
                vl_mse = sum(tmp_mses) / len(tmp_mses)
                vl_kld = sum(tmp_klds) / len(tmp_klds)

                tr_losses.append(tr_loss.item())
                vl_losses.append(vl_loss)

                # PRINT STEP
                if step % 30 == 1 :
                    # PRINT STEP
                    print("EPOCH {0} [step {1}] ::: TR {2:.4f} [ MSE {3:.4f}, KLD {4:.4f} ] | VL {5:.4f} [ MSE {6:.4f}, KLD {7:.4f} ]"
                          .format(epoch, step, tr_loss.item(), tr_mse.item(), tr_kld.item(), vl_loss, vl_mse, vl_kld))

                    # plot parameter histogram
                    plt.hist(mu.detach().cpu().flatten(), bins=100)
                    plt.savefig('./out/{}/expl/val/mu_hist.png'.format(self.config.model))
                    plt.close()
                    plt.hist(logvar.detach().cpu().flatten(), bins=100)
                    plt.savefig('./out/{}/expl/val/logvar_hist.png'.format(self.config.model))
                    plt.close()

                    # PLOT LOSSES
                    plt.plot(tr_losses)
                    plt.plot(vl_losses)
                    plt.title('Loss Curves')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.legend(['train loss','val loss'])
                    plt.savefig('./out/{}/img/loss_{}.png'.format(self.config.model, self.config.model_name))
                    plt.close()

                    # PLOT EXAMPLES
                    for i in range(10):
                        augs = model.augment(tr_X[i], 3)
                        plt.figure(figsize=(17,5))
                        plt.plot(tr_X[i].cpu().detach().numpy(), color='blue', linewidth=3)
                        for sample in augs :
                            plt.plot(sample.cpu().detach().numpy(), color='gray', linewidth=1)
                        plt.legend(['input','recon'])
                        plt.xlabel('time step')
                        plt.ylabel('CCY')
                        plt.title('Generated Example {}'.format(i+1))
                        plt.savefig('./out/{}/expl/train/expl_{}.png'.format(self.config.model, i+1))
                        plt.close()
                    for i in range(10):
                        augs = model.augment(vl_X[i], 3)
                        plt.figure(figsize=(17,5))
                        plt.plot(vl_X[i].cpu().detach().numpy(), color='blue', linewidth=3)
                        for sample in augs :
                            plt.plot(sample.cpu().detach().numpy(), color='gray', linewidth=1)
                        plt.legend(['input','recon'])
                        plt.xlabel('time step')
                        plt.ylabel('CCY')
                        plt.title('Generated Example {}'.format(i+1))
                        plt.savefig('./out/{}/expl/val/expl_{}.png'.format(self.config.model, i+1))
                        plt.close()

                    # SAVE MODEL & LOSSES
                    torch.save(model.state_dict(), './out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name))
                    with open('./out/{0}/loss/{0}_perf_idx_{1}.pkl'.format(self.config.model, self.config.model_name),'wb') as f :
                        dill.dump([tr_losses, vl_losses], f)
                    f.close()

                # EARLY STOP
                if model.start_std :
                    if self.config.no_early_stop:
                        converged = False
                    else:
                        converged = self.early_stop(vl_losses)
                else :
                    converged = False

                if converged:
                    break
            if converged:
                print('{} Model Converged...'.format(self.config.model.upper()))
                break
        torch.save(model.state_dict(), './out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name))
        self.test_vae(model)


    def test_vae(self, model):
        model.eval()
        tmp_losses = []
        for te_X, _ in self.te_data.dataloader :
            te_recon, mu, logvar = model(te_X, 99999)
            te_loss, te_mse, te_kld = model.criterion(te_recon, te_X, mu, logvar, 99999)
            tmp_losses.append(te_loss.item())
        te_loss = sum(tmp_losses) / len(tmp_losses)

        print("\n   converged model loss : {0:.4f}".format(te_loss))

    def train_ae(self, model):
        if self.config.reset_model :
            tr_losses, vl_losses = [], []
        else :
            model.load_state_dict(torch.load('./out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name)))
            with open('./out/{0}/loss/{0}_perf_idx_{1}.pkl'.format(self.config.model, self.config.model_name), 'rb') as f :
                tr_losses, vl_losses = dill.load(f)
            f.close()

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = model.criterion
        step = 0
        for epoch in range(1, self.config.epoch_num+1) :
            for tr_X, _ in self.tr_data.dataloader :
                step +=1

                # TRAIN
                model.train()
                tr_recon = model(tr_X)
                tr_loss = criterion(tr_recon, tr_X)
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                # VALIDATE
                model.eval()
                tmp_losses = []
                for vl_X, _ in self.vl_data.dataloader :
                    vl_recon = model(vl_X)
                    vl_loss = criterion(vl_recon, vl_X)
                    tmp_losses.append(vl_loss.item())
                vl_loss = sum(tmp_losses) / len(tmp_losses)

                tr_losses.append(tr_loss.item())
                vl_losses.append(vl_loss)

                # PRINT STEP
                if step % 30 == 1 :
                    # PRINT STEP
                    print("EPOCH {0} [step {1}] ::: TR {2:.4f} | VL {3:.4f}"
                          .format(epoch, step, 100*tr_loss.item(), 100*vl_loss))

                    # PLOT LOSSES
                    plt.plot(tr_losses)
                    plt.plot(vl_losses)
                    plt.title('Loss Curves')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.legend(['train loss','val loss'])
                    plt.savefig('./out/{}/img/loss_{}.png'.format(self.config.model, self.config.model_name))
                    plt.close()

                    # PLOT EXAMPLES
                    for i in range(10):
                        plt.figure(figsize=(17,5))
                        plt.plot(tr_X[i].cpu().detach().numpy(), color='blue', linewidth=3)
                        plt.plot(tr_recon[i].cpu().detach().numpy(), color='gray', linewidth=3)
                        plt.legend(['input','recon'])
                        plt.xlabel('time step')
                        plt.ylabel('CCY')
                        plt.title('Generated Example {}'.format(i+1))
                        plt.savefig('./out/{}/expl/train/expl_{}.png'.format(self.config.model, i+1))
                        plt.close()
                    for i in range(10):
                        plt.figure(figsize=(17,5))
                        plt.plot(vl_X[i].cpu().detach().numpy(), color='blue', linewidth=3)
                        plt.plot(vl_recon[i].cpu().detach().numpy(), color='gray', linewidth=3)
                        plt.legend(['input','recon'])
                        plt.xlabel('time step')
                        plt.ylabel('CCY')
                        plt.title('Generated Example {}'.format(i+1))
                        plt.savefig('./out/{}/expl/val/expl_{}.png'.format(self.config.model, i+1))
                        plt.close()

                    # SAVE MODEL & LOSSES
                    torch.save(model.state_dict(), './out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name))
                    with open('./out/{0}/loss/{0}_perf_idx_{1}.pkl'.format(self.config.model, self.config.model_name),'wb') as f :
                        dill.dump([tr_losses, vl_losses], f)
                    f.close()

                # EARLY STOP
                if self.config.no_early_stop:
                    converged = False
                else:
                    converged = self.early_stop(vl_losses)

                if converged:
                    break
            if converged:
                print('{} Model Converged...'.format(self.config.model.upper()))
                break
        torch.save(model.state_dict(), './out/{}/mdl/{}.mdl'.format(self.config.model, self.config.model_name))
        self.test_ae(model)

    def test_ae(self, model):
        model.eval()
        tmp_losses = []
        for te_X, _ in self.te_data.dataloader :
            te_recon = model(te_X)
            te_loss = model.criterion(te_recon, te_X)
            tmp_losses.append(te_loss.item())
        te_loss = sum(tmp_losses) / len(tmp_losses)

        print("\n   converged model loss : {0:.4f}".format(te_loss))

    def early_stop(self, losses):
        if len(losses) + 1 > self.config.min_epochs:
            if np.mean(losses[-100:]) < np.mean(losses[-40:]):
                return True
            else:
                return False
