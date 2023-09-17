import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config.config import config
from dataset.dataset import build_dataloader
from model.lightning import BuildLightningModel
from utils.tools import set_random_seed, check_model_name


def train(
        used_model: str = 'sscnn',
        cross_validation_fold_name: str = '',
):
    check_model_name(used_model)
    torch.set_float32_matmul_precision('high')
    set_random_seed(config['random_seed'])

    config['used_model'] = used_model

    # devices setting
    precision = config['precision']
    used_device = config['used_device']

    # load dataset
    train_dataloader = build_dataloader(config, mode='train', cross_val_name=cross_validation_fold_name)
    val_dataloader = build_dataloader(config, mode='val', cross_val_name=cross_validation_fold_name)

    # model init
    model = BuildLightningModel(
        model_name=config['used_model'],
        learn_rate=config['learn_rate'],
        cos_annealing_t_0=config['cos_annealing_t_0'],
        cos_annealing_t_mult=config['cos_annealing_t_mult'],
        cos_annealing_eta_min=config['cos_annealing_eta_min'],
        in_channel=config['in_channel'],
        spectrum_size=config['spectrum_size'],
        num_classes=len(config['type_list']),
        classes_name_list=config['type_list'],
        enable_torch_2=config['enable_torch_2.0'],
        torch_2_compile_mode=config['torch_2.0_compile_mode']
    )

    # log settings
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('[Info] Training start time: ', current_time)
    logger_list = []
    tensorboard_logger = TensorBoardLogger(save_dir=config['log_dir'], name='{}'.format(current_time))
    tensorboard_logger.log_hyperparams(config)
    logger_list.append(tensorboard_logger)
    if not config['debug'] and config['enable_wandb']:
        wandb_logger = WandbLogger(project=config['wandb_project_name'], save_dir=config['log_dir'],
                                   name='{}-{}'.format(config['used_model'], current_time))
        logger_list.append(wandb_logger)

    # early stopping
    early_stop_callback = EarlyStopping('val_acc', mode='max', min_delta=0.005, patience=40, verbose=True)

    # make checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['checkpoint_dir'], '{}'.format(current_time), 'checkpoints'),
        filename='best-{epoch}-{val_acc:.5f}',
        save_top_k=5,
        monitor='val_acc',
        mode='max',
        save_weights_only=False
    )

    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=used_device,
        precision=precision,
        logger=logger_list,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        max_epochs=config['epochs'],
        log_every_n_steps=1,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        fast_dev_run=config['debug'],
        # inference_mode=False,  # to avoid pl test error when using torch 2.0
    )
    # train
    trainer.fit(model, train_dataloader, val_dataloader)

    # test
    best_model_path = checkpoint_callback.best_model_path
    print('[Info] best model path: ', best_model_path)
    best_model = BuildLightningModel.load_from_checkpoint(best_model_path)
    best_model.eval()
    test_dataloader = build_dataloader(config, mode='test', cross_val_name=cross_validation_fold_name)
    trainer.test(best_model, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--used_model', type=str, default='sscnn')
    parser.add_argument('--cross_name', type=str, default='')
    opt = parser.parse_args()
    train(
        opt.used_model,
        opt.cross_name
    )
