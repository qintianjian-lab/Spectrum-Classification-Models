import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassAUROC

from utils.tools import check_model_name

try:
    from model.sscnn import SSCNN
    from model.c_net import CNET
    from model.rac_net import RACNET
    from model.rc_net import RCNET
    from model.convnext_1d import CONVNEXT1D
    from model.one_dim_cnn import ONEDIMCNN
except ImportError:
    raise ImportError('[Error] import model failed!')


class BuildLightningModel(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 learn_rate: float,
                 cos_annealing_t_0: int,
                 cos_annealing_t_mult: int,
                 cos_annealing_eta_min: float,
                 in_channel: int,
                 spectrum_size: int,
                 num_classes: int,
                 classes_name_list: list[str, ...],
                 label_smoothing: float = 0.,
                 enable_torch_2: bool = True,
                 torch_2_compile_mode: str = 'default'
                 ):
        super().__init__()
        assert torch_2_compile_mode in ['default', 'reduce-overhead', 'max-autotune'], \
            '[Error] torch_2_compile_mode must be in [default, reduce-overhead, max-autotune]'
        check_model_name(model_name)
        print('[Info] model_name: {}'.format(model_name))
        self.model = eval(model_name.upper())(
            in_channel=in_channel,
            out_channel=num_classes,
            spectrum_size=spectrum_size
        ) if not enable_torch_2 else torch.compile(
            eval(model_name.upper())(
                in_channel=in_channel,
                out_channel=num_classes,
                spectrum_size=spectrum_size
            ),
            mode=torch_2_compile_mode
        )
        if enable_torch_2:
            print('[Info] Using PyTorch 2.0 compile')
        self.learn_rate = learn_rate
        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.train_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.train_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.train_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)

        self.val_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.val_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.val_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.test_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.test_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)
        self.test_pred = []
        self.test_label = []
        self.classes_name_list = classes_name_list

        self.softmax = nn.Softmax(dim=1)
        self.best_val_acc = 0.0
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        spectrum, label, _ = batch
        pred = self.model(spectrum)
        loss = self.criterion(pred, label)
        self.log('train_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_precision(pred, label)
        self.log('train_precision', self.train_precision, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_recall(pred, label)
        self.log('train_recall', self.train_recall, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_auroc(pred, label)
        self.log('train_auroc', self.train_auroc, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        spectrum, label, _ = batch
        pred = self.model(spectrum)
        loss = self.criterion(pred, label)
        self.log('val_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.val_acc(pred, label)
        self.val_precision(pred, label)
        self.val_recall(pred, label)
        self.val_auroc(pred, label)
        return loss

    def on_validation_epoch_end(self):
        _val_acc = self.val_acc.compute()
        if _val_acc > self.best_val_acc:
            self.best_val_acc = _val_acc
            self.log('best_val_acc', self.best_val_acc, prog_bar=True, on_epoch=True)
        self.log('val_acc', _val_acc, prog_bar=True, on_epoch=True)
        self.val_acc.reset()

        _val_precision = self.val_precision.compute()
        self.log('val_precision', _val_precision, prog_bar=True, on_epoch=True)
        self.val_precision.reset()

        _val_recall = self.val_recall.compute()
        self.log('val_recall', _val_recall, prog_bar=True, on_epoch=True)
        self.val_recall.reset()

        _val_auroc = self.val_auroc.compute()
        self.log('val_auroc', _val_auroc, prog_bar=True, on_epoch=True)
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        spectrum, label, _ = batch
        pred = self.model(spectrum)
        loss = self.criterion(pred, label)
        self.log('test_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.test_acc(pred, label)
        self.test_precision(pred, label)
        self.test_recall(pred, label)
        self.test_auroc(pred, label)
        self.test_pred.extend(list(np.argmax(self.softmax(pred).cpu().numpy(), axis=1)))
        self.test_label.extend(list(label.cpu().numpy()))
        return loss

    def on_test_epoch_end(self):
        _test_acc = self.test_acc.compute()
        self.log('test_acc', _test_acc, prog_bar=True, on_epoch=True)
        self.test_acc.reset()

        _test_precision = self.test_precision.compute()
        self.log('test_precision', _test_precision, prog_bar=True, on_epoch=True)
        self.test_precision.reset()

        _test_recall = self.test_recall.compute()
        self.log('test_recall', _test_recall, prog_bar=True, on_epoch=True)
        self.test_recall.reset()

        _test_auroc = self.test_auroc.compute()
        self.log('test_auroc', _test_auroc, prog_bar=True, on_epoch=True)
        self.test_auroc.reset()

        cm = wandb.plot.confusion_matrix(
            preds=np.asarray(self.test_pred),
            y_true=np.asarray(self.test_label),
            probs=None,
            class_names=self.classes_name_list,
            title="Confusion Matrix in Test dataset",
        )
        wandb.log({"Confusion Matrix": cm})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learn_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cos_annealing_t_0,
            T_mult=self.cos_annealing_t_mult,
            eta_min=self.cos_annealing_eta_min
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'name': 'lr'
            },
        }
