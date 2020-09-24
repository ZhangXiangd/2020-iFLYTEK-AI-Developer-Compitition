import os, sys, glob, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import pandas as pd
import numpy as np
import time, datetime
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
# from resnext101_wsl import resnext101_32x8d_wsl

# from scheduler import GradualWarmupScheduler

import torch
# torch.cuda.empty_cache()
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset

print(torch.cuda.current_device())
print(torch.cuda.device_count())

# input dataset
train_jpg = np.array(glob.glob('./train/*/*.png'))
print('[Info]Training image numbers: ', train_jpg.shape[0])


# 自定义数据读取类，继承Dataset类
class MyDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # 每次读取一个数据
        img = Image.open(self.train_jpg[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # label: MCI:2, AD:1, CN:0
        if 'CN' in self.train_jpg[index]:
            label = 0
        elif 'AD' in self.train_jpg[index]:
            label = 1
        else:
            label = 2

        return img, torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.train_jpg)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        # efficientnet-b5
        model = EfficientNet.from_name('efficientnet-b5')
        model.load_state_dict(torch.load('./pretrained_models/efficientnet-b5-b6417697.pth'))
        in_ftrs = model._fc.in_features
        model._fc = nn.Linear(in_ftrs, 3)
        
        # ResNeXt101
        model = models.resnext101_32x8d(pretrained=False)
        model_path = os.path.join(os.getcwd(), 'pretrained_models/resnext101_32x8d-8ba56ff5.pth')
        pre = torch.load(model_path)
        model.load_state_dict(pre)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        
        # seresnext 101
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load('./pretrained_models/se_resnext101_32x4d-3b2fe3d8.pth'))
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        in_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(in_ftrs, 3)
        
        # resnext101_32x8d_wsl
        model = resnext101_32x8d_wsl(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)
        
        '''
        model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=None)
        # inceptionresnetv2和inceptionv4: 不修改为1001会报错
        model.last_linear = nn.Linear(in_features=1536, out_features=1001, bias=True)
        model.load_state_dict(torch.load('./pretrained_models/inceptionresnetv2-520b38e4.pth'))
        in_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(in_ftrs, 3)

        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


def train(train_loader, model, criterion, optimizer):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    model.train()

    for input, target in train_loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.sum().item()
        train_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return train_loss_sum / n, train_acc_sum / n


def validate(val_loader, model, criterion):
    val_loss_sum, val_acc_sum, n = 0.0, 0.0, 0
    model.eval()
    for input, target in val_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target.long())

        val_loss_sum += loss.sum().item()
        val_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return val_loss_sum / n, val_acc_sum / n

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

###########################################TRAIN######################################################
# random_state = 2020, 2021, 2022
# model = efficientnet-b5, seresnext101, inception-resnet-v2
skf = KFold(n_splits=5, random_state=2021, shuffle=True)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg)):
    print('Fold: ', fold_idx)
    train_loader = torch.utils.data.DataLoader(
        MyDataset(train_jpg[train_idx],
                  transforms.Compose([
                      transforms.Resize((299, 299)),
                      # transforms.Resize((512, 512)),
                      transforms.RandomRotation(90),  # TODO
                      transforms.ColorJitter(contrast=(1,2)), # TODO
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=20, shuffle=True, num_workers=8, pin_memory=True  # 8， 12， 20
    )

    val_loader = torch.utils.data.DataLoader(
        MyDataset(train_jpg[val_idx],
                  transforms.Compose([
                      transforms.Resize((299, 299)),
                      # transforms.Resize((512, 512)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=20, shuffle=False, num_workers=8, pin_memory=True
    )

    model = Net().cuda()
    # model = nn.DataParallel(model).cuda()
    criterion = LabelSmoothingCrossEntropy().cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    '''
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) 
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5,
                                              after_scheduler=scheduler)
    # this zero gradient update is needed to avoid a warning message
    optimizer.zero_grad()
    optimizer.step()
    '''

    best_loss = 1000
    best_acc = 0
    epochs = 70


    for epoch in range(epochs):
        # scheduler_warmup.step()
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)
        scheduler.step()
        print('LR: ', optimizer.param_groups[0]['lr'])


        print('%s | Epoch %d | train loss： %.4f | train acc： %.4f | val_loss： %.4f | val_acc： %.4f'
              % (time, epoch, train_loss, train_acc, val_loss, val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './inception_fold{0}_best_loss.pt'.format(fold_idx))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './inception_fold{0}_best_acc.pt'.format(fold_idx))


