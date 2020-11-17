import os
import time

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import *
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from data import EmbryoDataset, get_data
from model import *

writer = SummaryWriter()
iteration = 0


def main():
    data_path = r'/mnt/data2/like/data/embryo/cropped'
    phase = '脱水后'
    test_size = 0.2
    epochs = 80
    batch_size = 16
    lr = 0.0001

    # model loading
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = seresnet18(num_classes=2, pretrained=True)
    model = nn.DataParallel(model).cuda()

    # data loading
    image_paths, label = get_data(data_path, phase)
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(image_paths, label, test_size=test_size,
                                                                            random_state=1998, stratify=label)
    n_train, n_valid = len(train_paths), len(valid_paths)

    train_dl = DataLoader(
        EmbryoDataset(train_paths, train_labels,
                      transform=Compose([
                          RandomCrop(448),
                          RandomHorizontalFlip(),
                          RandomVerticalFlip(),
                          ColorJitter(),
                          ToTensor(),
                          Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
                      ])),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)
    valid_dl = DataLoader(
        EmbryoDataset(valid_paths, valid_labels,
                      transform=Compose([
                          CenterCrop(448),
                          ToTensor(),
                          Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
                      ])),
        batch_size=batch_size // 2, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True)

    # config optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # print info
    valid_pos = np.sum(valid_labels > 0)
    p = valid_pos / n_valid
    ce = -p * np.log(p) - (1 - p) * np.log(1 - p)
    print(f'''Starting training:
               Epochs:          {epochs}
               Batch size:      {batch_size}
               Learning rate:   {lr}
               Training size:   {n_train}
               Validation size: {n_valid}
               CrossEntropy:    {ce}
           ''')

    for epoch in range(epochs):
        # train for one epoch
        train(train_dl, model, criterion, optimizer, scheduler, epoch)
        scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % 1 == 0:
            acc = validate(valid_dl, model, criterion)

            # TODO: remember best accuracy and save checkpoint


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global writer, iteration
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        acc.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard
        writer.add_scalar('Loss/train', losses.val, iteration)
        writer.add_scalar('Accuracy/train', acc.val, iteration)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
        iteration += 1

        if i % 5 == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion):
    global writer, iteration
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        acc.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # tensorboard
    writer.add_scalar('Loss/test', losses.avg, iteration)
    writer.add_scalar('Accuracy/test', acc.avg, iteration)

    print(('\033[32mTesting Results: Accuracy {acc.avg:.3f} Loss {loss.avg:.5f}\033[0m'
           .format(acc=acc, loss=losses)))

    return acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
