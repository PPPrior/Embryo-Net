import logging
import os
import time

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import *
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import EmbryoDataset
from model import *


def loss_batch(model, loss_func, image, label, opt=None):
    image, label = image.cuda(), label.cuda()
    pred = model(image)
    loss = loss_func(pred, label)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters, 0.1)
        opt.step()

    return loss.item(), len(image)


def fit(model, data_path, epochs=50, batch_size=8, lr=0.001, val_percent=0.1):
    # get image & label
    dataset = EmbryoDataset(data_path)
    n_valid = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_valid
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid])
    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)
    valid_dl = DataLoader(dataset=valid_ds,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True)

    train_transform = Compose([
        RandomCrop(size=448),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225))
    ])

    test_transform = Compose([
        CenterCrop(size=448),
    ])

    print(f'''Starting training:
           Epochs:          {epochs}
           Batch size:      {batch_size}
           Learning rate:   {lr}
           Training size:   {n_train}
           Validation size: {n_valid}
       ''')

    writer = SummaryWriter()
    iteration = 0

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='image') as pbar:
            model.train()
            for image, label in train_dl:
                loss, _ = loss_batch(model, loss_func, image, label, optimizer)

                writer.add_scalar('Loss/train', loss, iteration)
                pbar.set_postfix(**{'loss (batch)': loss})
                pbar.update(image.shape[0])

                iteration += 1
                if iteration % (len(dataset) // (5 * batch_size)) == 0:
                    model.eval()
                    with torch.no_grad():
                        losses, nums = zip(
                            *[loss_batch(model, loss_func, image, label) for image, label in valid_dl]
                        )
                    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                    scheduler.step(val_loss)

                    writer.add_scalar('Loss/test', val_loss, iteration)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iteration)
                    logging.info('Validation cross entropy: {}'.format(val_loss))

    writer.close()


def main():
    model = nn.DataParallel(BCNN(num_classes=2)).cuda()
    path = r'/mnt/data2/like/data/embryo/medical'
    fit(model, path, lr=0.001, epochs=50, batch_size=4)

    prefix = './checkpoints/' + str(model.__class__) + '_'
    savepath = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
    torch.save(model.state_dict(), savepath)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # main()
    from model import seresnet50

    model = nn.DataParallel(seresnext50(pretrained=True)).cuda()
