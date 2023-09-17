import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from seaborn import heatmap
from torchmetrics.classification import MulticlassROC, MulticlassConfusionMatrix
from tqdm import tqdm

from config.config import config
from dataset.dataset import build_dataloader
from model.lightning import BuildLightningModel


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def interpolate_color(start, end, fraction):
    r1, g1, b1 = int(start[1:3], 16), int(start[3:5], 16), int(start[5:7], 16)
    r2, g2, b2 = int(end[1:3], 16), int(end[3:5], 16), int(end[5:7], 16)

    fraction = math.log10(fraction * 9 + 1) / math.log10(10)  # Applying log transformation

    r = round(r1 + (r2 - r1) * fraction)
    g = round(g1 + (g2 - g1) * fraction)
    b = round(b1 + (b2 - b1) * fraction)

    return f"#{r:02X}{g:02X}{b:02X}"


def predict(checkpoint_path: str, cross_name: str, save_path: str = './roc'):
    device_number = config['used_device'][0]
    # set device
    device = 'cuda:{}'.format(device_number) if torch.cuda.is_available() else 'cpu'
    print('[Info] Using device: ', device)
    torch.cuda.set_device(device_number)

    model = BuildLightningModel.load_from_checkpoint(os.path.join(checkpoint_path)).to(device)
    roc = MulticlassROC(num_classes=len(config['type_list'])).to(device)
    cm = MulticlassConfusionMatrix(num_classes=len(config['type_list'])).to(device)
    test_dataloader = build_dataloader(config, mode='test', cross_val_name=cross_name)
    model.eval()

    for batch in tqdm(test_dataloader, ncols=100):
        spec, label, _ = batch
        spec = spec.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model.model(spec)
        roc(pred, label)
        cm(pred, label)
    fpr, tpr, thresholds = roc.compute()
    roc.reset()
    pred_cm = cm.compute().cpu().numpy()
    cm.reset()
    # roc curve
    header = ['fpr', 'tpr', 'thresholds']
    create_dir(os.path.join(save_path))
    for _type, _fpr, _tpr, _thresholds in tqdm(zip(config['type_list'], fpr, tpr, thresholds), ncols=100):
        _fpr = _fpr.tolist()
        _tpr = _tpr.tolist()
        _thresholds = _thresholds.tolist()
        data = []
        for _f, _t, _th in zip(_fpr, _tpr, _thresholds):
            data.append([_f, _t, _th])
        df = pd.DataFrame(data=data, columns=header)
        df.to_csv(os.path.join(save_path, '{}.csv'.format(_type)), index=False)
    # confusion matrix and save as png image (figure size = (20, 20))
    fig = plt.figure(figsize=(20, 20))
    start_color = "#c9ddf0"
    end_color = "#08306b"
    num_colors = 150
    colors = [start_color]
    for i in range(1, num_colors):
        fraction = i / (num_colors - 1)
        interpolated_color = interpolate_color(start_color, end_color, fraction)
        colors.append(interpolated_color)
    colors.append(end_color)
    ax = heatmap(pred_cm, annot=True, fmt='d', cmap=colors, xticklabels=config['type_list'],
                 yticklabels=config['type_list'], square=True, linewidths=.1)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')
    fig.add_axes(ax)
    fig.savefig(os.path.join(save_path, 'confusion_matrix.pdf'))
    fig.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    fig.savefig(os.path.join(save_path, 'confusion_matrix.svg'))


if __name__ == '__main__':
    model_name = 'sscnn'
    chkpt = {
        'sscnn': 'Your Best Checkpoint.ckpt',
        'racnet': 'Your Best Checkpoint.ckpt',
        'convnext': 'Your Best Checkpoint.ckpt'}
    checkpoint = chkpt[model_name]
    cross_val_name = 'fold_2'
    predict(checkpoint, cross_val_name, save_path=f'roc_{model_name}')
