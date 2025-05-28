import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import *
import wandb
import numpy as np


class ATTACK:
    def __init__(self, source, target, num_classes, TARGETED=False):
        self.source = source
        self.target = target
        self.num_classes = num_classes
        self.iter = 0
        self.TARGETED = TARGETED

    def update_target(self, outputs_clean, y, counter):
        if self.TARGETED:
            self.target = 0
            self.target_label = (y[self.target] + 1) % self.num_classes
        # else:
        #     acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        #     while acc_target_be.item() == 0.:
        #         target += 1
        #         acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        #         if target > self.cfg.TEST.BATCH_SIZE - self.source - 1:
        #             target = 0
        #     self.target = target

        self.counter = counter

    def generate_attacks(
        self,
        sur_model,
        x,
        y,
        randomize=False,
        epsilon=16 / 255,
        alpha=2 / 255,
        num_iter=10,
    ):
        source = self.source
        target = self.target

        fixed = torch.zeros_like(
            x.clone()[:-source], requires_grad=False
        )  # benign samples # torch.Size([190, 3, 32, 32])

        if randomize:
            delta_0 = torch.rand_like(x[-source:])
        else:
            delta_0 = 127.5 / 255

        # adv = (torch.zeros_like(x.clone()[-source:]) - x[-source:] + delta_0).requires_grad_(True)  # malcious # torch.Size([10, 3, 32, 32])
        adv = (torch.zeros_like(x.clone()[-source:])).requires_grad_(True)  # malcious # torch.Size([10, 3, 32, 32])
        adv_pad = torch.cat((fixed, adv), 0)  # torch.Size([200, 3, 32, 32])

        if self.TARGETED:
            for t in tqdm(range(num_iter), disable=True):
                x_adv = x + adv_pad
                out = sur_model(x_adv)
                loss = nn.CrossEntropyLoss(reduction="none")(
                    out[target].reshape(1, -1), self.target_label.reshape(1)
                )
                loss.backward()

                self.iter += 1

                print(
                    "Learning Progress :%2.2f %% , loss1 : %f "
                    % ((t + 1) / num_iter * 100, loss.item()),
                    end="\r",
                )

                adv.data = (adv - alpha * adv.grad.detach().sign()).clamp(
                    -epsilon, epsilon
                )
                adv.data = (adv.data + x[-source:]).clamp(0, 1) - (x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0)
                adv.grad.zero_()
        else:
            # adv = torch.zeros_like(x.clone()[-source:], requires_grad=True)
            # adv_pad = torch.cat((fixed, adv), 0)  # torch.Size([200, 3, 32, 32])
            # random_std = torch.tensor(np.random.uniform(1.13, 1.20, 1), requires_grad=False).to(x.device)
            # random_mean = torch.tensor(np.random.uniform(0.0, 0.2, 1), requires_grad=False).to(x.device)
            # from torch.optim import SGD
            # optimizer = SGD([adv], lr=0.001, momentum=0.9)
            # x.requires_grad = False
            # for t in tqdm(range(100), disable=True):
            #     x_adv = x + adv_pad  # benign + initialize malicious sample (127.5 / 255)

            #     # Calculate the std of x_adv
            #     x_adv_std = torch.std(x_adv)
            #     x_adv_mean = torch.mean(x_adv)
            #     # print(x_adv_std)

            #     # Define a loss term to make x_adv's std close to the randomly selected std
            #     std_loss = ((x_adv_std - random_std) ** 2) * (10 ** 9) + ((random_mean - x_adv_mean) ** 2) * (10 ** 7)

            #     # Combine losses (if needed, adjust weights here)
            #     loss = std_loss
            #     # Zero gradients, perform a backward pass, and update the weights
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     # Clamp adv values to ensure they are within the allowed range
            #     # adv.data = adv.data.clamp(-epsilon, epsilon)
            #     # adv.data = (adv.data + x[-source:]).clamp(0, 1) - x[-source:]
            #     adv_pad.data = torch.cat((fixed, adv), 0)
            #     self.iter += 1
            #     # adv_pad.data = torch.zeros_like(adv_pad.data)
            #     # adv.grad.zero_()
            for t in tqdm(range(num_iter), disable=True):
                x_adv = x + adv_pad  # benign + initialize malcious sample (127.5 / 255)
                
                out = sur_model(x_adv)
                loss = nn.CrossEntropyLoss(reduction="none")(
                    out[:-source], y[:-source]
                ).clamp(min=0, max=5)
                # loss = -nn.CrossEntropyLoss(reduction="none")(
                #     out[:-source], self.target * torch.ones_like(y[:-source])
                # ).clamp(min=0, max=5)
                # loss = nn.CrossEntropyLoss(reduction="none")(
                #     out, y
                # ).clamp(min=0, max=5)
                loss = loss.sum()
                # loss += (- adv.data).sum()
                loss.backward()
                self.iter += 1
                # if loss.item() > 1:
                #     break

                # print(
                #     "Learning Progress :%2.2f %% , loss2 : %f "
                #     % ((t + 1) / num_iter * 100, loss.item()),
                #     end="\r",
                # )
                adv.data = (adv + alpha * adv.grad.detach().sign()).clamp(
                    -epsilon, epsilon
                )
                adv.data = (adv.data + x[-source:]).clamp(0, 1) - (x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0)
                adv.grad.zero_()
        # print(loss.item(), random_std.item(), x_adv_std.item())
        # print(loss.item())
        x_adv = x + adv_pad
        return x_adv

    def compute_acc(self, outputs_clean, outputs_adv, y):
        target = self.target
        source = self.source
        acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        acc_source_be = (outputs_clean.max(1)[1][-source:] == y[-source:]).float().sum()
        acc_clean = (outputs_clean.max(1)[1] == y).float().sum()
        acc_adv = (outputs_adv.max(1)[1] == y).float().sum()
        acc_target_af = (outputs_adv[target].argmax() == y[target]).float()
        acc_source_af = (outputs_adv.max(1)[1][-source:] == y[-source:]).float().sum()

        acc_benign_be = (outputs_clean.max(1)[1][:-source] == y[:-source]).float().sum()
        acc_benign_af = (outputs_adv.max(1)[1][:-source] == y[:-source]).float().sum()
        if self.cfg.ATTACK.TARGETED:
            acc_target_af = (outputs_adv[target].argmax() == self.target_label).float()

        return (
            acc_target_be,
            acc_target_af,
            acc_clean,
            acc_adv,
            acc_source_be,
            acc_source_af,
            acc_benign_be,
            acc_benign_af,
        )
