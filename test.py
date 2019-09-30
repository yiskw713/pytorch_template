import argparse
import csv
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torchvision
import yaml

from addict import Dict
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# from libs.loss_fn.myloss import MyLoss
# from libs.models.mymodel import MyModel
from libs.class_label_map import get_label2id_map
from libs.dataset import FlowersDataset
from libs.mean import get_mean, get_std
from libs.meter import AverageMeter
from libs.metric import accuracy
# from libs.transformer import MyTransformer


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for image classification with Flowers Recognition Dataset')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('mode', type=str, help='validation or test')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='path to the trained model. If you do not specify, the trained model, \
            \'best_acc1_model.prm\' in result directory will be used.'
    )

    return parser.parse_args()


def test(loader, model, n_classes, device):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # calculate confusion matrix
    c_matrix = np.zeros((n_classes, n_classes), dtype=np.int)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(loader):
            x = sample['img']
            t = sample['class_id']
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)

            # measure accuracy and record loss
            acc1 = accuracy(output, t, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            # keep predicted results and gts for calculate F1 Score
            _, pred = output.max(dim=1)
            gts += list(t.to("cpu").numpy())
            preds += list(pred.to("cpu").numpy())

            c_matrix += confusion_matrix(
                t.to("cpu").numpy(),
                pred.to('cpu').numpy(),
                labels=[i for i in range(n_classes)]
            )

    f1s = f1_score(gts, preds, average="macro")

    return top1.avg, f1s, c_matrix


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # cpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    # Dataloader
    data = FlowersDataset(
        CONFIG,
        transform=Compose([
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std())
        ]),
        mode='validation'
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    # the number of classes
    n_classes = len(get_label2id_map())

    if CONFIG.model == 'resnet18':
        print('ResNet18 will be used as a model.')
        model = torchvision.models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features=in_features,
            out_features=n_classes,
            bias=True
        )
    elif CONFIG.model == 'resnet34':
        print('ResNet34 will be used as a model.')
        model = torchvision.models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features=in_features,
            out_features=n_classes,
            bias=True
        )
    else:
        print('There is no model appropriate to your choice. '
              'You have to choose resnet18 or resnet34 as a model in config.yaml')
        sys.exit(1)

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(
            os.path.join(CONFIG.result_path, 'best_acc1_model.prm')
        )
    model.load_state_dict(state_dict)

    # train and validate model
    print('\n------------------------Start testing------------------------\n')

    # validation
    acc1, f1s, c_matrix = test(loader, model, n_classes, device)

    print(
        'acc1: {:.5f}\tF1 Score: {:.5f}'.format(acc1, f1s))

    df = pd.DataFrame(
        {'acc@1': [acc1], 'f1score': [f1s]},
        columns=['acc@1', 'f1score'],
        index=None
    )

    df.to_csv(
        os.path.join(CONFIG.result_path, '{}_log.csv').format(args.mode), index=False)

    with open(os.path.join(CONFIG.result_path, '{}_c_matrix.csv').format(args.mode), 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(c_matrix)


if __name__ == '__main__':
    main()
