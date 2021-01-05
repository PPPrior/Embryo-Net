import os
import time

from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from data import EmbryoDataset, get_data
from data.transforms import *
from utils import AverageMeter, accuracy
from model import *

writer = SummaryWriter()
iteration = 0
best_acc = 0


def main():
    global best_acc
    data_path = r'/mnt/data2/like/data/embryo/cropped'
    phase = '[1-6]h'  # in ('D[4-8]', '脱水后', '0h', '[1-6]h')
    test_size = 0.2
    epochs = 60
    batch_size = 16
    lr = 0.0001

    # model loading
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # model = seresnet18(num_classes=2, pretrained=True)
    model = Fusion('seresnet18', 'max')
    model = nn.DataParallel(model).cuda()

    # data loading
    # image_paths, label = get_data(data_path, phase)
    # TODO: fusion
    image_paths, label = get_data(data_path)
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(image_paths, label, test_size=test_size,
                                                                            random_state=1998, stratify=label)
    n_train, n_valid = len(train_paths), len(valid_paths)

    train_dl = DataLoader(
        EmbryoDataset(train_paths, train_labels,
                      # transform=Compose([
                      #     RandomCrop(488),
                      #     RandomHorizontalFlip(),
                      #     RandomVerticalFlip(),
                      #     ColorJitter(),
                      #     ToTensor(),
                      #     # Normalize((0.485, 0.456, 0.406),
                      #     #           (0.229, 0.224, 0.225)),
                      #     # Normalize((0.501, 0.436, 0.233),
                      #     #           (0.126, 0.113, 0.094)),
                      # ])
                      # transform=Compose([
                      #     GroupCenterCrop(488),
                      #     Stack(),
                      #     ToTorchFormatTensor(),
                      #     GroupNormalize((0.501, 0.436, 0.233),
                      #                    (0.126, 0.113, 0.094)),
                      # ])
                      ),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)
    valid_dl = DataLoader(
        EmbryoDataset(valid_paths, valid_labels,
                      # transform=Compose([
                      #     CenterCrop(488),
                      #     ToTensor(),
                      #     # Normalize((0.485, 0.456, 0.406),
                      #     #           (0.229, 0.224, 0.225)),
                      #     # Normalize((0.501, 0.436, 0.233),
                      #     #           (0.126, 0.113, 0.094)),
                      # ])
                      # transform=Compose([
                      #     GroupCenterCrop(488),
                      #     Stack(),
                      #     ToTorchFormatTensor(),
                      #     GroupNormalize((0.501, 0.436, 0.233),
                      #                    (0.126, 0.113, 0.094)),
                      # ])
                      ),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    # config optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 170], gamma=0.5)
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

    cm, pred = None, None
    for epoch in range(epochs):
        # train for one epoch
        train(train_dl, model, criterion, optimizer, epoch)
        scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % 1 == 0:
            acc, rst, p = validate(valid_dl, model, criterion)

            if acc >= best_acc:
                best_acc = acc
                pred = [np.argmax(x) for x in rst]
                cm = confusion_matrix(valid_labels, pred)
            # TODO: remember best accuracy and save checkpoint

    evaluate(cm, p, valid_paths, valid_labels, pred)


def train(train_loader, model, criterion, optimizer, epoch):
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
    rst, p = [], []
    m = torch.nn.Softmax(dim=1)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            rst.extend(output.cpu().numpy())
            p.extend(m(output))
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

    return acc.avg, rst, p


def evaluate(cm, ps, path, target, pred):
    print("Confusion matrix")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * precision * recall / (precision + recall)
    sensitivity = tp / (fn + tp)
    specificity = tn / (tn + fp)

    print(f'''
Accuracy:   {accuracy * 100:.2f}%
Precision:  {precision * 100:.2f}%
Recall:     {recall * 100:.2f}%
F1 score:   {f1 * 100:.2f}%
Sensitivity:{sensitivity * 100:.2f}%
Specificity:{specificity * 100:.2f}%
    ''')

    for i, t in enumerate(target):
        print(f'{i}, {t}, {pred[i]}, {ps[i].cpu().numpy()}, {path[i][0]}')


if __name__ == '__main__':
    main()
