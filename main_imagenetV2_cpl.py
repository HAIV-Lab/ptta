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

imagenet_v_mask = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11,110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122,123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135,136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148,149, 15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 16, 160,161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173,174, 175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186,187, 188, 189, 19, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 2, 20, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 21, 210,211, 212, 213, 214, 215, 216, 217, 218, 219, 22, 220, 221, 222, 223,224, 225, 226, 227, 228, 229, 23, 230, 231, 232, 233, 234, 235, 236,237, 238, 239, 24, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 25, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 26, 260, 261,262, 263, 264, 265, 266, 267, 268, 269, 27, 270, 271, 272, 273, 274,275, 276, 277, 278, 279, 28, 280, 281, 282, 283, 284, 285, 286, 287,288, 289, 29, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 3, 30, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 31, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 32, 320, 321, 322, 323, 324,325, 326, 327, 328, 329, 33, 330, 331, 332, 333, 334, 335, 336, 337,338, 339, 34, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 35,350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 36, 360, 361, 362,363, 364, 365, 366, 367, 368, 369, 37, 370, 371, 372, 373, 374, 375,376, 377, 378, 379, 38, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 39, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 4, 40,400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 41, 410, 411, 412,413, 414, 415, 416, 417, 418, 419, 42, 420, 421, 422, 423, 424, 425,426, 427, 428, 429, 43, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 44, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 45, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 46, 460, 461, 462, 463,464, 465, 466, 467, 468, 469, 47, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 48, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 49, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 5, 50, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 51, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 52, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 53, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 54, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 55, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 56, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 57, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 58, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 59, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 6, 60, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 61, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 62, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 63, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 64, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 65, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 66, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 67, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 68, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 69, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 7, 70, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 71, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 72, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 73, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 74, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 75, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 76, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 77, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 78, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 79, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 8, 80, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 81, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 82, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 83, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 84, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 85, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 86, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 87, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 88, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 89, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 9, 90, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 91, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 92, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 93, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 94, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 95, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 96, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 97, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 98, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 99, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
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
            # images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(ids, images)[:, imagenet_v_mask]
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
        from models.wide_resnet import WideResNet
        subnet = WideResNet(widen_factor=2, depth=40, num_classes=10)
        checkpoint = torch.load('models/Hendrycks2020AugMixWRN.pt')
        subnet.load_state_dict(checkpoint)
        print("use wrn40")

    # set learning rate
    args.learning_rate = args.learning_rate if args.batch_size == 64 else args.learning_rate * args.batch_size / 64
    # subnet.load_state_dict(init)
    subnet = subnet.cuda()

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if args.dataset == 'cifar10c':
        args.e_margin = math.log(10)*0.40
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
    common_corruptions = ['domain_shift']
    # common_corruptions = ['motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    # common_corruptions = ['brightness','pixelate','gaussian_noise','motion_blur','zoom_blur','glass_blur','impulse_noise','jpeg_compression','defocus_blur','elastic_transform','shot_noise','frost','snow','fog','contrast',]
    print(common_corruptions)

    if args.algorithm == 'tent':
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = tent.Tent(subnet, optimizer)
        # adapt_model = subnet
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
    elif args.algorithm == 'ptta_eta2':
        import utils.ptta_eata2 as ptta_eata2
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_eata2.PTTA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    fisher_alpha=args.fisher_alpha, neighbor=args.neighbor)
        print("the model is ptta_eata2")
    elif args.algorithm == 'ptta_eata':
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

    elif args.algorithm == 'ptta_deyo':
        import utils.ptta_deyo as ptta_deyo
        subnet = ptta_deyo.configure_model(subnet)
        params, param_names = ptta_deyo.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_deyo.PTTA(subnet, optimizer, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor)
        print("the model is ptta_deyo")
    elif args.algorithm == 'ptta_cpl':
        import utils.ptta_cpl as ptta_cpl
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = ptta_cpl.CPL(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor)
        print("the model is ptta_cpl")
    elif args.algorithm == 'cpl':
        import utils.cpl as cpl
        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = cpl.CPL(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, \
                    loss2_weight=args.loss2_weight, queue_size=args.queue_size, \
                    neighbor=args.neighbor)
        print("the model is cpl")

    else:
        assert False, NotImplementedError

    for corrupt in common_corruptions:
        if args.exp_type == 'each_shift_reset':
            print("reset")
            # adapt_model.reset()
        elif args.exp_type == 'continual':
            print("continue")
        else:
            assert False, NotImplementedError

        args.corruption = corrupt
        print(args.corruption)

        validdir = args.data_corruption
        print('Test on %s' %validdir)
        from dataset.selectedRotateImageFolder import SelectedRotateImageFolderwithID, te_transforms
        val_dataset = SelectedRotateImageFolderwithID(validdir, te_transforms, original=False, rotation=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, 
                                                        num_workers=args.workers, pin_memory=True)

        val_dataset.switch_mode(True, False)
        adapt_model.init_mb(val_loader)

        # val_dataset, val_loader = prepare_test_data(args)
        # val_dataset.switch_mode(True, False)

        # for _ in range(10):
        #     top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        print(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        if args.algorithm in ['eata', 'eta']:
            print(f"num of reliable samples is {adapt_model.num_samples_update_1}, num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
            adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0