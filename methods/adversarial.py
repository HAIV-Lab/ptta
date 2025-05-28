from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import foolbox as fb
from torch.nn.functional import cross_entropy

class Adversarial(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        bounds = (-2.11890, 2.641)
        self.fmodel = fb.PyTorchModel(model, bounds=bounds, device='cuda')

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.fmodel)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def kl_divergence(p, q):
    """Calculate KL divergence between two distributions."""
    p = torch.softmax(p, dim=1)
    q = torch.softmax(q, dim=1)
    return torch.sum(p * torch.log(p / q), dim=1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, fmodel):
    """Forward and adapt model on batch of data.

    Measure dependency between original and adversarial predictions,
    take gradients, and update params if dependency is high enough.
    """
    # forward
    outputs = model(x)
    
    # adversarial attack
    attack = fb.attacks.FGSM()
    epsilons = 2 / 255
    labels = outputs.argmax(dim=1)
    adversarial_images = attack(fmodel, x, labels, epsilons=epsilons)[1]
    outputs_2 = model(adversarial_images)

    # compute KL divergence
    kl_div = kl_divergence(outputs, outputs_2)
    
    # define threshold
    threshold = 0.2  # 定义依赖程度的阈值
    # threshold = 1
    
    # create mask based on KL divergence
    mask = kl_div < threshold
    # mask = softmax_entropy(outputs_2) < threshold
    
    # apply mask to outputs
    masked_outputs = outputs[mask]
    # compute cross entropy loss
    # loss = softmax_entropy(outputs).mean(0)
    loss = softmax_entropy(masked_outputs).mean(0)
    # adapt
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
