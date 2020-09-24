import os, sys, glob, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import pandas as pd
import numpy as np
import time, datetime
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

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

# print(torch.cuda.current_device())
# print(torch.cuda.device_count())


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
    def __init__(self, name):
        super(Net, self).__init__()
        self.name = name

        if self.name == 'efficient':
            # efficientnet-b5
            model = EfficientNet.from_name('efficientnet-b5')
            in_ftrs = model._fc.in_features
            model._fc = nn.Linear(in_ftrs, 3)
        elif self.name == 'seresnext':
            # seresnext 101
            model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
            model.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
            in_ftrs = model.last_linear.in_features
            model.last_linear = nn.Linear(in_ftrs, 3)
        else:
            model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=None)
            in_ftrs = model.last_linear.in_features
            model.last_linear = nn.Linear(in_ftrs, 3)

        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out

def predict(test_loader, model, tta=5):
    model.eval()

    # test_pred_tta = None
    test_pred_tta = np.zeros((test_jpg.shape[0], 3), dtype=np.float32)
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)  # 按行叠加数组

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta / tta


def inference(saved_model_path, net, resize, batch_size):
    test_pred = np.zeros((test_jpg.shape[0], 3), dtype=np.float32)
    for model_path in saved_model_path:
        print(model_path)
        test_loader = torch.utils.data.DataLoader(
            MyDataset(test_jpg,
                      transforms.Compose([
                          # transforms.Resize((299, 299)),
                          transforms.Resize(resize),
                          transforms.RandomAffine(10),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

        model = net.cuda()
        model.load_state_dict(torch.load(model_path))
        # model = nn.DataParallel(model).cuda()
        if test_pred is None:
            test_pred = predict(test_loader, model, tta=5)
        else:
            test_pred += predict(test_loader, model, tta=5)

    return test_pred / len(saved_model_path)

if __name__ == "__main__":
    test_jpg = ['./test/AD&CN&MCI/{0}.jpg'.format(x) for x in range(1, 2001)]
    test_jpg = np.array(test_jpg)
    print('[Info] Test image numbers: ', test_jpg.shape[0])

    efficient_path = ['efficientb5_fold0.pt','efficientb5_fold1.pt','efficientb5_fold2.pt',
                      'efficientb5_fold3.pt','efficientb5_fold4.pt','efficientb5_fold5.pt']
    seresnext_path = ['seresnext101_fold0_best_acc.pt','seresnext101_fold0_best_loss.pt',
                      'seresnext101_fold1_best_acc.pt','seresnext101_fold1_best_loss.pt',
                      'seresnext101_fold2_best_acc.pt','seresnext101_fold2_best_loss.pt',
                      'seresnext101_fold3_best_acc.pt','seresnext101_fold3_best_loss.pt',
                      'seresnext101_fold4_best_acc.pt','seresnext101_fold4_best_loss.pt']
    inception_path = ['inception_fold0_best_acc.pt','inception_fold0_best_loss.pt',
                      'inception_fold1_best_acc.pt','inception_fold1_best_loss.pt',
                      'inception_fold2_best_acc.pt','inception_fold2_best_loss.pt',
                      'inception_fold3_best_acc.pt','inception_fold3_best_loss.pt',
                      'inception_fold4_best_acc.pt','inception_fold4_best_loss.pt']

    efficient_pred = inference(saved_model_path=efficient_path, net=Net(name='efficient'), resize=(512, 512), batch_size=8)
    np.save("efficient_pred.npy", efficient_pred)
    print(efficient_pred.shape)
    seresnext_pred = inference(saved_model_path=seresnext_path, net=Net(name='seresnext'), resize=(512, 512), batch_size=12)
    np.save("seresnext_pred.npy", seresnext_pred)
    print(seresnext_pred.shape)
    inception_pred = inference(saved_model_path=inception_path, net=Net(name='inception'), resize=(299, 299), batch_size=20)
    np.save("inception_pred.npy", inception_pred)
    print(inception_pred.shape)
    ensemble_pred = (efficient_pred + seresnext_pred + inception_pred) / 3

    test_csv = pd.DataFrame()
    test_csv['uuid'] = list(range(1, 2001))
    test_csv['label'] = np.argmax(ensemble_pred, 1)
    test_csv['label'] = test_csv['label'].map({2: 'MCI', 1: 'AD', 0: 'CN'})
    test_csv.to_csv('ensamble.csv', index=False)
