import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
import datetime
import math
from sklearn.metrics import confusion_matrix
from mislas import *

def unwrap_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m


class Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight

        self.model = model

        # ---- 1) Freeze all parameters ----
        for p in self.model.parameters():
            p.requires_grad = False

        # ---- 2) Unwrap model and unfreeze fc only ----
        self.real_model = unwrap_model(self.model)

        if args.methods == 'GLMC':
            for p in self.real_model.fc.parameters():
                p.requires_grad = True
            for p in self.real_model.fc_cb.parameters():
                p.requires_grad = True
        elif args.methods == 'MISLAS':
            for p in self.real_model.fc.parameters():
                p.requires_grad = True
            if not getattr(args, 'use_samn', 1):
                # MISLAS baseline: also train the learnable weight scaling
                for p in self.real_model.learned_norm.parameters():
                    p.requires_grad = True
        else:
            for p in self.real_model.fc.parameters():
                p.requires_grad = True

        # ---- Optional: set BatchNorm to eval to freeze running stats ----
        if args.methods == 'MISLAS':
            for m in self.real_model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
        else:
            for m in self.real_model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()

        params_to_update = [p for p in self.model.parameters() if p.requires_grad]

        # self.optimizer = torch.optim.SGD(self.model.fc.parameters(), momentum=0.9, lr=self.lr,weight_decay=args.weight_decay)
        self.optimizer = torch.optim.SGD(params_to_update, momentum=0.9, lr=self.lr,
                                         weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        # self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=int(self.epochs / 2.))
        self.log = log
        self.beta = args.beta
        self.update_weight()
        if self.args.methods == 'MISLAS':
            self.train_criterion = LabelAwareSmoothing(cls_num_list=self.cls_num_list, smooth_head=args.smooth_head,
                                                       smooth_tail=args.smooth_tail).cuda()
        else:
            self.train_criterion = nn.CrossEntropyLoss()


    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def train(self):
        best_acc1 = 0
        p_losses = []
        pv_losses = []
        for epoch in range(self.start_epoch, self.epochs):
            torch.cuda.synchronize()
            est = time.perf_counter()
            print('lr={}'.format(self.optimizer.param_groups[0]['lr']))
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            p_loss = 0.

            for i, (inputs, targets) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                # measure data loading time
                data_time.update(time.time() - end)

                if self.args.methods == 'GLMC':

                    _, output, __, ___ = self.model(input_org_1.cuda(), train=True)
                    loss = self.train_criterion(output, target_org.cuda())

                else:
                    output = self.model(input_org_1.cuda(), train=True)
                    loss = self.train_criterion(output, target_org.cuda())

                losses.update(loss.item(), inputs[0].size(0))
                p_loss += loss.item()

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
            torch.cuda.synchronize()
            eet = time.perf_counter()
            print('epoch time', eet - est)

            a_loss = p_loss / len(self.train_loader)
            p_losses.append(a_loss)
            acc1, av_loss = self.validate(epoch=epoch)
            pv_losses.append(av_loss)
            self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

        return p_losses, pv_losses

    def horizontal_flip_aug(self, model):
        def aug_model(inputs, train=False):
            logits = model(inputs, train=train)
            h_logits = model(inputs.flip(3), train=train)
            return (logits + h_logits) / 2

        return aug_model

    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        model = self.horizontal_flip_aug(self.model)
        all_preds = []
        all_targets = []

        pv_loss = 0.

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input, train=False)
                loss = self.train_criterion(output, target.cuda())

                pv_loss += loss.item()

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1, top5=top5))

            apv_loss = pv_loss / len(self.val_loader)
            self.log.info(output)

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg, apv_loss

    def SimSiamLoss(self,p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self,optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
