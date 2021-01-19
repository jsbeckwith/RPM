import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utility import PGMdataset, RAVENdataset, ToTensor
from hrinet import HriNet

""" GPUID = '2,5,6'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID """

parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='HriNet')
parser.add_argument('--dataset', type=str, default='Balanced-RAVEN', choices=['PGM', 'Balanced-RAVEN'])
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=str, default='save/')
parser.add_argument('--Balanced_RAVEN_path', type=str, default='../Balanced-RAVEN/70k_dataset')
parser.add_argument('--save', type=str, default='save/')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_beta', type=float, default=0.0)
parser.add_argument('--cuda', default=True )

args = parser.parse_args()
torch.cuda.manual_seed(args.seed)
args.test_figure_configurations = [0,1,2,3,4,5,6]
test = RAVENdataset(args.Balanced_RAVEN_path, "test", args.test_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]))
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)
model = HriNet(args)
model.load_model(args.resume, 179)
print('Loaded model')
pmodel = torch.nn.DataParallel(model)
torch.backends.cudnn.benchmark = True
pmodel = pmodel.cuda()

def test():
    model.eval()
    accuracy = 0
    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, meta_target) in enumerate(testloader):
            counter += 1
            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()   
            output = pmodel(image)
            pred = output[0].data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum().numpy()
            accuracy = correct * 100. / target.size()[0]   
            acc = accuracy
            acc_all += acc
            if batch_idx % 100 == 0:
                print("interim accuracy, batch", batch_idx, ":")
                print((acc_all / float(counter)))
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():
    avg_test_acc = test()

if __name__ == '__main__':
    main()
