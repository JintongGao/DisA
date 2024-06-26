import random
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet_cifar_ot 
from imbalance_data.imbalance_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10
from losses import LDAMLoss
from opts import parser
import warnings
from util import *
import torch.nn.functional as F
import numpy as np

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

def main():
    args = parser.parse_args()
    if args.dataset == 'cifar100':
        num_classes = 100
        model = getattr(resnet_cifar_ot, 'resnet32_fe_c100')()
        feat_len = 128
        use_norm = True if args.loss_type == 'LDAM' else False
        classifier_etf = getattr(resnet_cifar_ot, 'ETF_Classifier')(feat_in=feat_len, num_classes=100)
        classifier = getattr(resnet_cifar_ot, 'Classifier')(feat_in=feat_len, num_classes=100, use_norm=use_norm)

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = IMBALANCECIFAR100(root=args.root, imb_factor=args.imb_factor,
                                          rand_number=args.rand_number, train=True, download=True)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.CIFAR100(root=args.root, train=False, download=True, transform=transform_val)

    elif args.dataset == 'cifar10':
        num_classes = 10
        model = getattr(resnet_cifar_ot, 'resnet32_fe')()
        feat_len = 64 
        use_norm = True if args.loss_type == 'LDAM' else False
        classifier_etf = getattr(resnet_cifar_ot, 'ETF_Classifier')(feat_in=feat_len, num_classes=10)
        classifier = getattr(resnet_cifar_ot, 'Classifier')(feat_in=feat_len, num_classes=10, use_norm=use_norm)

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = IMBALANCECIFAR10(root=args.root, imb_factor=args.imb_factor,
                                          rand_number=args.rand_number, train=True, download=True)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.CIFAR10(root=args.root, train=False, download=False, transform=transform_val)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    train_cls_num_list = np.array(cls_num_list)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
        classifier_etf = classifier_etf.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier_etf = torch.nn.DataParallel(classifier_etf).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            classifier.load_state_dict(checkpoint['classifier'])
            #classifier_etf.ori_M=checkpoint['classifier_etf']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.train_rule == 'None':
        per_cls_weights = None
    elif args.train_rule == 'DRW':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    else:
        warnings.warn('Sample rule is not listed')

    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
    else:
        warnings.warn('Loss type is not listed')
        return

    flag = 'val'

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            feat = model(input)
            output = classifier(feat)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)

        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20
        print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
              float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
              float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))


if __name__ == '__main__':
    main()
