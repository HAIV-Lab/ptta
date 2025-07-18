"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from .ptta_eata_id import FeatureBank, compute_gradient
import torch.nn.functional as F

class PTTA(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000),
                 queue_size=1000, neighbor=1, loss2_weight=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        args = {'filter_ent': 1,'aug_type': 'patch', 'occlusion_size':112,'row_start':56,'column_start':56, "deyo_margin": 0.5*math.log(1000), "margin_e0": 0.4*math.log(1000), "filter_plpd": 1, "plpd_threshold": 0.3, "reweight_ent": 1, "reweight_plpd": 1, 'patch_len':4}
        from utils.dict2class import Dict2Class
        args = Dict2Class(args)
        if model.__class__.__name__.lower() == 'visiontransformer':
            args.plpd_threshold = 0.2
        elif model.__class__.__name__.lower() == 'resnet':
            args.plpd_threshold = 0.3
        # else:
        #     raise Exception("Model not supported")
        print('the args of deyo are:', args)
        self.args = args
        # if args.wandb_log:
        #     import wandb
        self.steps = steps
        self.episodic = episodic
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
        self.memory_bank = FeatureBank(queue_size, neighbor)
        self.alpha = 1 / (neighbor + 1)
        self.loss2_weight = loss2_weight

        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def forward(self, x, iter_=0, targets=None, flag=False, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = self.forward_and_adapt_deyo(x, iter_, self.args,
                                                                              self.deyo_margin,
                                                                              self.margin_e0, targets, group)
                else:
                    outputs = self.forward_and_adapt_deyo(x, iter_, self.args,
                                                    self.deyo_margin,
                                                    self.margin_e0, targets, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = self.forward_and_adapt_deyo(x, iter_, 
                                                                                                    self.args, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, group)
                else:
                    outputs = self.forward_and_adapt_deyo(x, iter_, 
                                                    self.args, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, group, self)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None
        # self.memory_bank.reset()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_deyo(self, x, iter_, args, deyo_margin, margin, targets=None, group=None):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = self.model(x)
        grads = compute_gradient(outputs.clone().detach())
        
        entropys = softmax_entropy(outputs)
        if args.filter_ent:
            filter_ids_1 = torch.where((entropys < deyo_margin))
        else:    
            filter_ids_1 = torch.where((entropys <= math.log(1000)))
        entropys = entropys[filter_ids_1]
        backward = len(entropys)
        if backward==0:
            if targets is not None:
                return outputs
            return outputs

        x_prime = x[filter_ids_1]
        x_prime = x_prime.detach()
        if args.aug_type=='occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
            x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
        elif args.aug_type=='patch':
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
            x_prime = resize_o(x_prime)
        elif args.aug_type=='pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        with torch.no_grad():
            outputs_prime = self.model(x_prime)
        
        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
        
        if args.filter_plpd:
            filter_ids_2 = torch.where(plpd > args.plpd_threshold)
        else:
            filter_ids_2 = torch.where(plpd >= -2.0)
        entropys = entropys[filter_ids_2]
        # if outputs[filter_ids_1][filter_ids_2].shape[0] > 0:
        #     self.memory_bank.update(x[filter_ids_1][filter_ids_2], grads[filter_ids_1][filter_ids_2].clone().detach(), outputs[filter_ids_1][filter_ids_2].clone().detach())
        final_backward = len(entropys)
        
        if targets is not None:
            corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
            
        if final_backward==0:
            del x_prime
            del plpd
            
            if targets is not None:
                return outputs
            return outputs
            
        plpd = plpd[filter_ids_2]
        
        if targets is not None:
            corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

        if args.reweight_ent or args.reweight_plpd:
            coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                    args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                    )            
            entropys = entropys.mul(coeff)

        if self.memory_bank.num_features_stored > 0:
            pred_labels, probs, img, grads_m = self.memory_bank.refine_predictions(grads.clone().detach())
            alpha = self.alpha
            x = x * alpha + img.cuda() * (1 - alpha)
            probs = outputs.softmax(dim=-1) * alpha + probs.cuda() * (1 - alpha)
            outputs2 = self.model(x)
            loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='batchmean')

        loss = entropys.mean(0)
        if self.memory_bank.num_features_stored > 0:
            loss += loss2 * self.loss2_weight
        if final_backward != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        del x_prime
        del plpd
        
        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

