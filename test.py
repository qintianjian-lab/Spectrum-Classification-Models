import os
from typing import Tuple

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
from tqdm import tqdm

from config.config import config
from dataset.dataset import build_dataloader
from model.lightning import BuildLightningModel


def load_model(model_path: str) -> torch.nn.Module:
    try:
        model = BuildLightningModel.load_from_checkpoint(os.path.join(model_path)).model
        # remove model. prefix
        state_dict = {k.replace('model.', ''): v for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict)
        print('[Info] Load model from {}'.format(model_path))
    except Exception as e:
        print('[Error] Load model failed, error message: {}'.format(e))
        exit(1)
    return model


def inference(config: dict, model_path: str, cross_val_dataset_name: str, device: torch.device) -> Tuple[
    float, float, float]:
    dataloader = build_dataloader(config, mode='test', cross_val_name=cross_val_dataset_name)
    model = load_model(model_path).to(device)
    model.eval()
    # init metrics
    accuracy = MulticlassAccuracy(num_classes=len(config['type_list'])).to(device)
    f1_score = MulticlassF1Score(num_classes=len(config['type_list'])).to(device)
    auroc = MulticlassAUROC(num_classes=len(config['type_list'])).to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            spectrum, label, data_id = batch
            spectrum = spectrum.to(device)
            label = label.to(device)
            output = model(spectrum)
            # metrics
            accuracy(output, label)
            f1_score(output, label)
            auroc(output, label)
        _accuracy = accuracy.compute().item()
        _f1_score = f1_score.compute().item()
        _auroc = auroc.compute().item()
        print('[Info] For cross validation dataset: {}'.format(cross_val_dataset_name))
        print('[Info] Accuracy: {:.4f}'.format(_accuracy))
        print('[Info] F1 score: {:.4f}'.format(_f1_score))
        print('[Info] AUROC: {:.4f}'.format(_auroc))
        # release memory
        del model
        return _accuracy, _f1_score, _auroc


if __name__ == '__main__':
    model_save_path = {
        'kfold_0': './',
        'kfold_1': './',
        'kfold_2': './',
        'kfold_3': './',
        'kfold_4': './',
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    avg_accuracy = 0
    avg_f1_score = 0
    avg_auroc = 0
    acc_list = []
    f1_list = []
    auroc_list = []
    for cross_val_dataset_name, model_path in model_save_path.items():
        _acc, _f1, _auroc = inference(config, model_path, cross_val_dataset_name, device)
        avg_accuracy += _acc
        avg_f1_score += _f1
        avg_auroc += _auroc
        acc_list.append(_acc)
        f1_list.append(_f1)
        auroc_list.append(_auroc)
    avg_accuracy /= len(model_save_path)
    avg_f1_score /= len(model_save_path)
    avg_auroc /= len(model_save_path)
    print('=' * 100)
    print('[Info] Average accuracy: {:.4f}'.format(avg_accuracy))
    print('[Info] Average F1 score: {:.4f}'.format(avg_f1_score))
    print('[Info] Average AUROC: {:.4f}'.format(avg_auroc))
    print('[Info] Accuracy list: {}'.format(acc_list))
    print('[Info] F1 score list: {}'.format(f1_list))
    print('[Info] AUROC list: {}'.format(auroc_list))
    # save as result.txt
    with open('result.txt', 'w') as f:
        f.write('Average accuracy: {:.4f}\n'.format(avg_accuracy))
        f.write('Average F1 score: {:.4f}\n'.format(avg_f1_score))
        f.write('Average AUROC: {:.4f}\n'.format(avg_auroc))
        f.write('Accuracy list: {}\n'.format(['{:.4f}'.format(i) for i in acc_list]))
        f.write('F1 score list: {}\n'.format(['{:.4f}'.format(i) for i in f1_list]))
        f.write('AUROC list: {}\n'.format(['{:.4f}'.format(i) for i in auroc_list]))
