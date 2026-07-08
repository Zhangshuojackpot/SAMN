import sys
import os
import time
import argparse
import torch
import numpy as np
import random
from torch.backends import cudnn
from utils import util
from utils.util import *
from model import ResNet_cifar_stage2 as ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
import logging
from Stage2_Trainer import Trainer
from mislas import *

# train set
parser = argparse.ArgumentParser(description="SAMN: Stage 2 Classifier Fine-tuning with Monotonic Norm Constraints")
parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--root', type=str, default='./data/', help="dataset setting")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=('resnet18', 'resnet32', 'resnet34', 'resnet50', 'resnext50_32x4d'))
parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--beta', type=float, default=0.5, help="augment mixture")
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0, type=float, metavar='W',
                    help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')

parser.add_argument('--methods', default='CE', type=str, help='weighted for Loss')
parser.add_argument('--resample_weighting', default=0., type=float, help='weighted for sampling probability (q(1,k))')
parser.add_argument('--label_weighting', default=1.2, type=float, help='weighted for Loss')
parser.add_argument('--contrast_weight', default=4, type=int, help='Mixture Consistency  Weights')
# etc.
parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=
'./SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--root_log', type=str, default='SAMN-CVPR2026/fineoutput/')
parser.add_argument('--root_model', type=str, default='SAMN-CVPR2026/fineoutput/')
parser.add_argument('--store_name', type=str, default='SAMN-CVPR2026/fineoutput/')

parser.add_argument('--imagenet_traintxt', default='./data/data_txt/ImageNet_LT_train.txt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--imagenet_testtxt', default='./data/data_txt/ImageNet_LT_test.txt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--smooth_head', default=0.3, type=float, help='Mixture Consistency  Weights')
parser.add_argument('--smooth_tail', default=0.1, type=float, help='Mixture Consistency  Weights')
parser.add_argument('--use_samn', default=1, type=int,
                    help='1=PAVALinear head (SAMN), 0=standard Linear head (MISLAS baseline)')


best_acc1 = 0

def validate(model,val_loader,args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output

            output = model(input, train=False)


            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            output = 'Testing:  ' + str(i) + ' Prec@1:  ' + str(top1.val) + ' Prec@5:  ' + str(top5.val)
            print(output, end="\r")
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(flag='val', top1=top1, top5=top5))
        print(output)

def get_model(args, list):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_LT.resnext50_32x4d(args=args, class_counts=list, num_classes=args.num_classes)
        return net
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(args=args, class_counts=list, num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(args=args, class_counts=list, num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(args=args, class_counts=list, num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(args=args, class_counts=list, num_class=args.num_classes)
        return net

def get_dataset(args):
    transform_train,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=args.root)
        print("load cifar10")
        return trainset,testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return trainset,testset

    if args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.imagenet_traintxt, util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.imagenet_testtxt, transform_val)
        return trainset,testset

    if args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt,util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return trainset,testset

def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for PyTorch experiment reproducibility."""
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (single/multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic  # determinism and performance are mutually exclusive

    print(f"[INFO] Random seed set to {seed}, deterministic={deterministic}")

def main():
    args = parser.parse_args()
    print(args)
    args.store_name = '#'.join(["dataset: " + args.dataset, "arch: " + args.arch,"imbanlance_rate: " + str(args.imbanlance_rate), 'methods:' + args.methods, 'use_samn:' + str(args.use_samn)])
    prepare_folders(args)
    if args.seed is not None:
        set_seed(args.seed)
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    # Data loading code
    train_dataset,val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes

    cls_num_list = train_dataset.get_per_class_num()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),num_workers=args.workers, persistent_workers=True,pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, persistent_workers=True,pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    train_sampler = None
    weighted_train_loader = None

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args, train_cls_num_list)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()


    # optionally resume from a checkpoint
    # test from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("=> loaded checkpoint '{}'".format(args.resume))
        print("Testing started!")
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    #weighted_loader
    if args.methods == 'GLMC':
        args.lr = 0.00005
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers, persistent_workers=True,pin_memory=True,sampler=weighted_sampler)
    elif args.methods == 'MISLAS':
        args.lr = 0.0005
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, persistent_workers=True,
                                                            pin_memory=True, sampler=None, shuffle=True)
    else:
        if 'cifar' in args.dataset:
            args.lr = 0.0005
        if 'ImageNet-LT' in args.dataset:
            args.lr = 0.000005
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, persistent_workers=True,
                                                            pin_memory=True, sampler=None, shuffle=True)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list,log=logging)
    print('lr={}'.format(args.lr))
    p_losses, pv_losses = trainer.train()
    filename = '%s/%s/' % (args.root_model, args.store_name)
    np.savetxt(os.path.join(filename, 'p_losses.txt'), np.array(p_losses))
    np.savetxt(os.path.join(filename, 'pv_losses.txt'), np.array(pv_losses))
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    main()
