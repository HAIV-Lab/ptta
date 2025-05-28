import torch
import torch.nn as nn
from torchattacks.attack import Attack


class FGSM(nn.Module):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, eps=16 / 255, if_reverse=False, targeted=False):
        super().__init__()
        self.eps = eps
        self.if_reverse = if_reverse
        self.targeted = targeted
    
    def _get_perturbed_image(self, images, grad, step_size):
        if self.if_reverse:
            perturbed_image = images - step_size * grad
        else:
            perturbed_image = images + step_size * grad
        return torch.clamp(perturbed_image, min=0, max=1)

    def _is_correctly_classified(self, model, images, labels):
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            return accuracy >= 0.8

    def forward(self, images, labels, model, steps=None):
        r"""
        Overridden.
        """
        # images = images.clone().detach().to(images.device)
        # labels = labels.clone().detach().to(images.device)

        # images.requires_grad = True
        # outputs = model(images)
        # loss = nn.CrossEntropyLoss()
        
        # if self.targeted:
        #     raise ValueError("This method does not support targeted attacks.")
        
        # cost = loss(outputs, labels)
        # grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        # low, high = 1e-7, self.eps
        # while high - low > 1e-5:  # precision threshold
        #     mid = (low + high) / 2
        #     adv_images = self._get_perturbed_image(images, grad, mid)
        #     if self._is_correctly_classified(model, adv_images, labels):
        #         low = mid
        #     else:
        #         high = mid

        # # The maximum step size that does not cause misclassification
        # max_step = low
        # adv_images = self._get_perturbed_image(images, grad, max_step)
        # return adv_images

        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(images.device)

        if self.targeted:
            target_labels = model(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = model(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]
        if self.if_reverse:
            if steps is not None:
                adv_images = images - steps * grad
            else:
                adv_images = images - self.eps * grad
        else:
            if steps is not None:
                adv_images = images + steps * grad
            else:
                adv_images = images + self.eps * grad
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
