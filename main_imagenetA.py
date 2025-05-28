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

thousand_k_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 2, 14: -1, 15: 3, 16: -1, 17: 4, 18: -1, 19: -1, 20: -1, 21: -1, 22: 5, 23: 6, 24: -1, 25: -1, 26: -1, 27: 7, 28: -1, 29: -1, 30: 8, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 9, 38: -1, 39: 10, 40: -1, 41: -1, 42: 11, 43: -1, 44: -1, 45: -1, 46: -1, 47: 12, 48: -1, 49: -1, 50: 13, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 14, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 15, 71: 16, 72: -1, 73: -1, 74: -1, 75: -1, 76: 17, 77: -1, 78: -1, 79: 18, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 19, 90: 20, 91: -1, 92: -1, 93: -1, 94: 21, 95: -1, 96: 22, 97: 23, 98: -1, 99: 24, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 25, 106: -1, 107: 26, 108: 27, 109: -1, 110: 28, 111: -1, 112: -1, 113: 29, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 30, 125: 31, 126: -1, 127: -1, 128: -1, 129: -1, 130: 32, 131: -1, 132: 33, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 34, 144: 35, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 36, 151: 37, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 38, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 39, 235: 40, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 41, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 42, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 43, 284: -1, 285: -1, 286: -1, 287: 44, 288: -1, 289: -1, 290: -1, 291: 45, 292: -1, 293: -1, 294: -1, 295: 46, 296: -1, 297: -1, 298: 47, 299: -1, 300: -1, 301: 48, 302: -1, 303: -1, 304: -1, 305: -1, 306: 49, 307: 50, 308: 51, 309: 52, 310: 53, 311: 54, 312: -1, 313: 55, 314: 56, 315: 57, 316: -1, 317: 58, 318: -1, 319: 59, 320: -1, 321: -1, 322: -1, 323: 60, 324: 61, 325: -1, 326: 62, 327: 63, 328: -1, 329: -1, 330: 64, 331: -1, 332: -1, 333: -1, 334: 65, 335: 66, 336: 67, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 68, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 69, 362: -1, 363: 70, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 71, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 72, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 73, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 74, 398: -1, 399: -1, 400: 75, 401: 76, 402: 77, 403: -1, 404: 78, 405: -1, 406: -1, 407: 79, 408: -1, 409: -1, 410: -1, 411: 80, 412: -1, 413: -1, 414: -1, 415: -1, 416: 81, 417: 82, 418: -1, 419: -1, 420: 83, 421: -1, 422: -1, 423: -1, 424: -1, 425: 84, 426: -1, 427: -1, 428: 85, 429: -1, 430: 86, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 87, 438: 88, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 89, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 90, 457: 91, 458: -1, 459: -1, 460: -1, 461: 92, 462: 93, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 94, 471: -1, 472: 95, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 96, 484: -1, 485: -1, 486: 97, 487: -1, 488: 98, 489: -1, 490: -1, 491: -1, 492: 99, 493: -1, 494: -1, 495: -1, 496: 100, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: 101, 515: -1, 516: 102, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 103, 529: -1, 530: 104, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 105, 540: -1, 541: -1, 542: 106, 543: 107, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 108, 550: -1, 551: -1, 552: 109, 553: -1, 554: -1, 555: -1, 556: -1, 557: 110, 558: -1, 559: -1, 560: -1, 561: 111, 562: 112, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 113, 570: -1, 571: -1, 572: 114, 573: 115, 574: -1, 575: 116, 576: -1, 577: -1, 578: -1, 579: 117, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 118, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 119, 607: 120, 608: -1, 609: 121, 610: -1, 611: -1, 612: -1, 613: -1, 614: 122, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 123, 627: 124, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 125, 641: 126, 642: 127, 643: 128, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 129, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 130, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 131, 678: -1, 679: -1, 680: -1, 681: -1, 682: 132, 683: -1, 684: 133, 685: -1, 686: -1, 687: 134, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 135, 702: -1, 703: -1, 704: 136, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 137, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 138, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 139, 747: -1, 748: -1, 749: 140, 750: -1, 751: -1, 752: 141, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 142, 759: -1, 760: -1, 761: -1, 762: -1, 763: 143, 764: -1, 765: 144, 766: -1, 767: -1, 768: 145, 769: -1, 770: -1, 771: -1, 772: -1, 773: 146, 774: 147, 775: -1, 776: 148, 777: -1, 778: -1, 779: 149, 780: 150, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 151, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 152, 793: -1, 794: -1, 795: -1, 796: -1, 797: 153, 798: -1, 799: -1, 800: -1, 801: -1, 802: 154, 803: 155, 804: 156, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 157, 814: -1, 815: 158, 816: -1, 817: -1, 818: -1, 819: -1, 820: 159, 821: -1, 822: -1, 823: 160, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 161, 832: -1, 833: 162, 834: -1, 835: 163, 836: -1, 837: -1, 838: -1, 839: 164, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 165, 846: -1, 847: 166, 848: -1, 849: -1, 850: 167, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 168, 860: -1, 861: -1, 862: 169, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 170, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 171, 880: 172, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 173, 889: -1, 890: 174, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 175, 898: -1, 899: -1, 900: 176, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 177, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 178, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 179, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 180, 933: 181, 934: 182, 935: -1, 936: -1, 937: 183, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 184, 944: -1, 945: 185, 946: -1, 947: 186, 948: -1, 949: -1, 950: -1, 951: 187, 952: -1, 953: -1, 954: 188, 955: -1, 956: 189, 957: 190, 958: -1, 959: 191, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 192, 972: 193, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 194, 981: 195, 982: -1, 983: -1, 984: 196, 985: -1, 986: 197, 987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}
indices_in_1k = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]

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
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(images)[:, indices_in_1k]
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
    elif args.algorithm == 'noadapt':
        adapt_model = subnet
        adapt_model = adapt_model.cuda()
        adapt_model.eval()
        print("no adaptation")
    elif args.algorithm == 'sar':
        import utils.sar as sar
        from utils.sam import SAM
        net = sar.configure_model(subnet)
        params, param_names = sar.collect_params(net)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.learning_rate, momentum=0.9)
        # adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
        adapt_model = sar.SAR(net, optimizer, margin_e0=math.log(1000)*0.40)
    elif args.algorithm == 'cotta':
        import utils.cotta as cotta
        model = cotta.configure_model(subnet)
        params, param_names = cotta.collect_params(model)
        optimizer = torch.optim.SGD(params,
                   lr=args.learning_rate,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=0.0,
                   nesterov=True)
        adapt_model = cotta.CoTTA(model, optimizer,
                            steps=1,
                            episodic=False)

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

        val_dataset, val_loader = prepare_test_data(args)
        val_dataset.switch_mode(True, False)

        # for _ in range(10):
        #     top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        print(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        if args.algorithm in ['eata', 'eta']:
            print(f"num of reliable samples is {adapt_model.num_samples_update_1}, num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
            adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0