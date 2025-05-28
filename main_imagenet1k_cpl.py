from logging import debug
import os
import time
import argparse
import json
import random
import math
import torchvision, timm
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
from dataset.selectedRotateImageFolder import prepare_test_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tent
import eata
import utils.ptta_eata as ptta
import utils.ptta_tent as ptta_tent
import models.Res as Resnet



def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            ids, images, target = dl[0], dl[1], dl[2]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(ids, images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 3000 == 0:
            progress.display(i)
    return top1.avg, top5.avg


def get_args():

    parser = argparse.ArgumentParser(description='TTA Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='/apdcephfs/private_huberyniu/etta_exps/camera_ready_debugs', help='the output directory of this experiment')
    parser.add_argument('--dataset', default='imagenet', help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rotation', default=False, type=bool, help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # model name, support resnets
    parser.add_argument('--arch', default='resnet50', type=str, help='the default model architecture')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    

    # overall experimental settings
    parser.add_argument('--exp_type', default='continual', type=str, help='continual or each_shift_reset') 
    # 'cotinual' means the model parameters will never be reset, also called online adaptation; 
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--algorithm', default='eta', type=str, help='eata or eta or tent')  
    parser.add_argument('--learning_rate', default=0.00025, type=float, help='learning rate for the first stage of PTTA')
    parser.add_argument('--network', default='resnet50', type=str, help='the network architecture')

    # ptta settings
    parser.add_argument('--loss2_weight', default=3, type=float, help='the weight of the second loss in PTTA')
    parser.add_argument('--queue_size', default=1000, type=int, help='the size of the queue in PTTA')
    parser.add_argument('--neighbor', default=1, type=int, help='the number of neighbors in PTTA')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    seed = args.seed
    print(f"seed is {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.network == 'resnet50':
        print("use resnet50")
        subnet = Resnet.__dict__[args.arch](pretrained=True)
    # subnet = torchvision.models.resnet50(pretrained=True)
    elif args.network == 'vit':
        print("use vit")
        subnet = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif args.network == 'wrn40':
        from robustbench.utils import load_model
        # from models.wide_resnet import WideResNet
        # subnet = WideResNet(widen_factor=2, depth=40, num_classes=10)
        # checkpoint = torch.load('models/Hendrycks2020AugMixWRN.pt')
        # subnet.load_state_dict(checkpoint)
        subnet = load_model(model_name='Hendrycks2020AugMix_WRN', model_dir = "./ckpt", dataset='cifar100', threat_model='corruptions')
        print("use wrn40")
    elif args.network == 'wrn28':
        from robustbench.utils import load_model
        # subnet = load_model('Standard', '/data/lhl/lhl_code/tent/ckpt',
        #                "cifar10", "corruptions").cuda()
        subnet = load_model(model_name='Hendrycks2020AugMix_WRN', model_dir = "./ckpt", dataset='cifar100', threat_model='corruptions')
        print("use Hendrycks2020AugMix_WRN")
    else:
        assert False, NotImplementedError
    # set learning rate
    args.learning_rate = args.learning_rate if args.batch_size == 64 else args.learning_rate * args.batch_size / 64
    # subnet.load_state_dict(init)
    subnet = subnet.cuda()

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if args.dataset == 'cifar10c':
        args.e_margin = math.log(10)*0.40
        args.d_margin = 0.4
    if args.dataset == "cifar100c":
        args.e_margin = math.log(100)*0.40
        args.d_margin = 0.4
    
    print(args)

    # if args.exp_type == 'continual':
    #     common_corruptions = [[item, 'original'] for item in common_corruptions]
    #     common_corruptions = [subitem for item in common_corruptions for subitem in item]
    # elif args.exp_type == 'each_shift_reset':
    #     print("continue")
    # else:
    #     assert False, NotImplementedError
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    common_corruptions = ['original']
    # common_corruptions = ['frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if args.dataset == 'imagenet-r':
        common_corruptions = ['domain_shift']
    # common_corruptions = ['motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    # common_corruptions = ['brightness','pixelate','gaussian_noise','motion_blur','zoom_blur','glass_blur','impulse_noise','jpeg_compression','defocus_blur','elastic_transform','shot_noise','frost','snow','fog','contrast',]
    print(common_corruptions)

    if args.algorithm == 'tent':
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = tent.Tent(subnet, optimizer)
    elif args.algorithm == 'eta':
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = eata.EATA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'eata':
        # compute fisher informatrix
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)

        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        print("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = eata.EATA(subnet, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'ptta_eta':
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        print("the model is ptta_eta")
    elif args.algorithm == 'ptta_eta3':
        import utils.ptta_eata3 as ptta_eata3
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata3.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        print("the model is ptta_eta3")
    elif args.algorithm == 'ptta_eta_image':
        import utils.ptta_eata_image_feature as ptta_eata_image
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata_image.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor, image=True)
        print("the model is ptta_eata2")
    elif args.algorithm == 'ptta_eta_feature':
        import utils.ptta_eata_image_feature as ptta_eata_feature
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata_feature.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor, image=False)
        print("the model is ptta_eata2")
    elif args.algorithm == 'ptta_eta_logit':
        import utils.ptta_eata_image_feature as ptta_eata_logit
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata_logit.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor, image=False, logit=True)
        print("the model is ptta_eata2")
    elif args.algorithm == 'ptta_eata':
        # compute fisher informatrix
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        if args.dataset != 'cifar10c' and args.dataset != 'cifar100c':
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        adapt_model.fishers = fishers
        print("the model is ptta_eata")
    elif args.algorithm == 'ptta_eata2':
        # compute fisher informatrix
        import utils.ptta_eata2 as ptta_eata2
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)

        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata2.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        adapt_model.fishers = fishers
        print("the model is ptta_eata_v2")
    elif args.algorithm == 'ptta_tent':
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_tent.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        print("the model is ptta_tent")
    elif args.algorithm == 'ptta_tent_image':
        import utils.ptta_tent_image_feature as ptta_tent_image
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_tent_image.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor, image=True)
        print("the model is ptta_tent")
    elif args.algorithm == 'ptta_tent_feature':
        import utils.ptta_tent_image_feature as ptta_tent_feature
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_tent_feature.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor, image=False)
        print("the model is ptta_tent")
    elif args.algorithm == 'deyo':
        import utils.deyo as deyo
        subnet = deyo.configure_model(subnet)
        params, param_names = deyo.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = deyo.DeYO(subnet, optimizer)
        print("the model is deyo")

    elif args.algorithm == 'ptta_deyo':
        import utils.ptta_deyo as ptta_deyo
        subnet = ptta_deyo.configure_model(subnet)
        params, param_names = ptta_deyo.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_deyo.PTTA(subnet, optimizer, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor)
        print("the model is ptta_deyo")
    elif args.algorithm == 'ptta_deyo_image':
        import utils.ptta_deyo_image_feature as ptta_deyo_image
        subnet = ptta_deyo.configure_model(subnet)
        params, param_names = ptta_deyo.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_deyo_image.PTTA(subnet, optimizer, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor, image=True)
        print("the model is ptta_deyo")
    elif args.algorithm == 'ptta_deyo_feature':
        import utils.ptta_deyo_image_feature as ptta_deyo_feature
        subnet = ptta_deyo.configure_model(subnet)
        params, param_names = ptta_deyo.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_deyo_feature.PTTA(subnet, optimizer, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor, image=False)
        print("the model is ptta_deyo")
    elif args.algorithm == 'cpl':
        import utils.cpl as cpl
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = cpl.CPL(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
        print("the model is cpl")
    elif args.algorithm == 'ptta_cpl':
        import utils.ptta_cpl as ptta_cpl
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_cpl.CPL(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor)

    else:
        assert False, NotImplementedError

    for corrupt in common_corruptions:
        if args.exp_type == 'each_shift_reset':
            print("reset")
            adapt_model.reset()
        elif args.exp_type == 'continual':
            print("continue")
        else:
            assert False, NotImplementedError

        args.corruption = corrupt
        print(args.corruption)

        validdir = os.path.join(args.data, 'val')
        print('Test on %s' %validdir)
        from dataset.selectedRotateImageFolder import SelectedRotateImageFolderwithID, te_transforms
        val_dataset = SelectedRotateImageFolderwithID(validdir, te_transforms, original=False, rotation=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, 
                                                        num_workers=args.workers, pin_memory=True)

        val_dataset.switch_mode(True, False)
        adapt_model.init_mb(val_loader)

        # for _ in range(10):
        #     top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        print(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        if args.algorithm in ['eata', 'eta']:
            print(f"num of reliable samples is {adapt_model.num_samples_update_1}, num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
            adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0