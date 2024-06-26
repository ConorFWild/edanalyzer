from pathlib import Path

from rich import print as rprint
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt
import yaml
import numpy as np
# import tables
import zarr
from numcodecs import Blosc, Delta
import pandas as pd


from .resnet import resnet18, resnet10
from .simple_autoencoder import SimpleConvolutionalEncoder, SimpleConvolutionalDecoder

from edanalyzer.losses import categorical_loss


annotation_dtype = [
    ('epoch', '<i4'),
    ('meta_idx', '<i4'),
    ('decoy_idx', '<i4'),
    ('y', '<f4'),
    ('y_hat', '<f4'),
    ('rmsd', '<f4'),
    ('corr', '<f4'),
    ('system', '<U32'),
    ('dtag', '<U32'),
    ('event_num', 'i8')
]

class LitBuildScoring(lt.LightningModule):
    def __init__(self, output_dir):
        super().__init__()
        # self.automatic_optimization = False
        self.resnet = resnet10(num_classes=2, num_input=1, headless=True).float()
        # self.z_encoder = SimpleConvolutionalEncoder(input_layers=2)
        self.z_encoder = resnet10(num_classes=2, num_input=3, headless=True).float()
        self.x_encoder = SimpleConvolutionalEncoder(input_layers=1)
        # self.mol_encoder = SimpleConvolutionalEncoder(input_layers=1)
        self.mol_encoder = resnet10(num_classes=2, num_input=1, headless=True).float()
        self.mol_decoder = SimpleConvolutionalDecoder()
        self.x_decoder = SimpleConvolutionalDecoder(input_layers=512)
        self.z_decoder = SimpleConvolutionalDecoder(input_layers=512)
        self.mol_to_weight = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Sequential(

            nn.Linear(512,2),

        )
        self.train_annotations = []
        self.test_annotations = []

        self.output = output_dir

    def forward(self, z,):
        # mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)

        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)

        score = F.softmax(self.fc(z_encoding))

        # score = F.sigmoid(self.fc(z_encoding))

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-1, weight_decay=1e-2)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, cooldown=10)
        # return [optimizer], [lr_scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "test_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            },
        }
        # return optimizer

    def training_step(self, train_batch, batch_idx):
        (meta_idx, decoy_idx, embedding_idx, system, dtag, event_num), z, m, rmsd, corr, y = train_batch
        y = y.view(y.size(0), -1)

        # mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)

        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)

        # score = F.sigmoid(self.fc(z_encoding))
        # loss_1 = F.mse_loss(score, y)
        score = F.softmax(self.fc(z_encoding))
        loss = categorical_loss(score, y)

        # total_loss = loss_1

        self.log('train_loss', loss)


        for j in range(len(meta_idx)):
            self.train_annotations.append(
                {
                    "meta_idx": int(meta_idx[j].to(torch.device("cpu")).detach().numpy()),
                    "decoy_idx": int(decoy_idx[j].to(torch.device("cpu")).detach().numpy()),
                    # "y": float(y[j].to(torch.device("cpu")).detach().numpy()),
                    # "y": 0.1 * float(np.argmax(y[j].cpu().detach())),
                    # "y_hat": float(score[j]).to(torch.device("cpu")).detach().numpy()),
                    # "y_hat": 0.1 * float(np.argmax(score[j].cpu().detach())),
                    "y": float(y[j][1].cpu().detach()),
                    "y_hat": float(score[j][1].cpu().detach()),
                    'rmsd': float(rmsd[j].to(torch.device("cpu")).detach().numpy()),
                    'corr': float(corr[j].to(torch.device("cpu")).detach().numpy()),

                    "system": str(system[j]),
                    "dtag": str(dtag[j]),
                    "event_num": int(event_num[j])
                }
            )
        # self.annotations[]
        return loss

    def validation_step(self, test_batch, batch_idx):
        (meta_idx, decoy_idx, embedding_idx, system, dtag, event_num), z, m, rmsd, corr, y = test_batch
        y = y.view(y.size(0), -1)

        # mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)

        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)

        # print(f'Z Encoding: {z_encoding[0,:10]}')
        # print(f'Mol Encoding: {mol_encoding[0,:10]}')

        # score = F.sigmoid(self.fc(z_encoding))
        score = F.softmax(self.fc(z_encoding))
        loss = categorical_loss(score, y)


        # loss = F.mse_loss(score, y)

        self.log('test_loss', loss)

        for j in range(len(meta_idx)):
            self.test_annotations.append(
                {
                    "meta_idx": int(meta_idx[j].to(torch.device("cpu")).detach().numpy()),
                    "decoy_idx": int(decoy_idx[j].to(torch.device("cpu")).detach().numpy()),
                    # "y": float(y[j].to(torch.device("cpu")).detach().numpy()),
                    # "y": 0.1 * float(np.argmax(y[j].cpu().detach())),
                    # "y_hat": float(score[j].to(torch.device("cpu")).detach().numpy()),
                    # "y_hat": 0.1 * float(np.argmax(score[j].cpu().detach())),
                    "y": float(y[j][1].cpu().detach()),
                    "y_hat": float(score[j][1].cpu().detach()),
                    'rmsd': float(rmsd[j].to(torch.device("cpu")).detach().numpy()),
                    'corr': float(corr[j].to(torch.device("cpu")).detach().numpy()),
                    "system": str(system[j]),
                    "dtag": str(dtag[j]),
                    "event_num": int(event_num[j])
                }
            )

    def on_train_epoch_end(self):
        # Log the predictions
        predictions = self.train_annotations

        # Get the table file
        store_file = self.output / 'annotations.zarr'

        # If store: Load the store and group
        if store_file.exists():
            root = zarr.open(store_file, mode='a')
            train_annotation_table = root['train_annotations']

        # Else: Create store and create groups
        else:
            root = zarr.open(store_file, mode='a')
            train_annotation_table = root.create_dataset(
                'train_annotations',
                shape=(0,),
                chunks=(1000,),
                dtype=annotation_dtype,
                compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
            )
            test_annotation_table = root.create_dataset(
                'test_annotations',
                shape=(0,),
                chunks=(1000,),
                dtype=annotation_dtype,
                compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
            )

        # Append annotations

        annotations = np.array(
            [
                (
                    int(self.trainer.current_epoch),
                    int(_annotation['meta_idx']),
                    int(_annotation['decoy_idx']),
                    float(_annotation['y']),
                    float(_annotation['y_hat']),
                    float(_annotation['rmsd']),
                    float(_annotation['corr']),
                    str(_annotation['system']),
                    str(_annotation['dtag']),
                    int(_annotation['event_num'])
                )
                for _annotation in self.train_annotations
            ],
            dtype=annotation_dtype
        )
        train_annotation_table.append(annotations)

        self.train_annotations.clear()

    def on_validation_epoch_end(self):
        # Log the predictions
        predictions = self.test_annotations


        # Get the table file
        store_file = self.output / 'annotations.zarr'

        # If store: Load the store and group
        if store_file.exists():
            root = zarr.open(store_file, mode='a')
            test_annotation_table = root['test_annotations']

        # Else: Create store and create groups
        else:
            root = zarr.open(store_file, mode='a')
            train_annotation_table = root.create_dataset(
                'train_annotations',
                shape=(0,),
                chunks=(1000,),
                dtype=annotation_dtype,
                compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
            )
            test_annotation_table = root.create_dataset(
                'test_annotations',
                shape=(0,),
                chunks=(1000,),
                dtype=annotation_dtype,
                compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
            )

        # Append annotations

        annotations = np.array(
            [
                (
                    int(self.trainer.current_epoch),
                    int(_annotation['meta_idx']),
                    int(_annotation['decoy_idx']),
                    float(_annotation['y']),
                    float(_annotation['y_hat']),
                    float(_annotation['rmsd']),
                    float(_annotation['corr']),
                    str(_annotation['system']),
                    str(_annotation['dtag']),
                    int(_annotation['event_num'])
                )
                for _annotation in self.test_annotations
            ],
            dtype=annotation_dtype
        )
        test_annotation_table.append(annotations)

        #
        # _epoch = epoch
        # best_df = df[(df['epoch'] == _epoch)]
        # best_df = pd.DataFrame.from_records(annotations)
        # pr = []
        # for cutoff in np.linspace(0.0, 1.0, num=1000):
        #     true_hit_best_df = best_df[best_df['y'] > 0.9]
        #     negative_best_df = best_df[best_df['y'] <= 0.1]
        #     true_hit_df = true_hit_best_df[true_hit_best_df['y_hat'] > cutoff]
        #     false_hit_df = negative_best_df[negative_best_df['y_hat'] > cutoff]
        #     true_negative_df = negative_best_df[negative_best_df['y_hat'] <= cutoff]
        #     false_negative_df = true_hit_best_df[true_hit_best_df['y_hat'] <= cutoff]
        #     tp = len(true_hit_df)
        #     fp = len(false_hit_df)
        #     tn = len(true_negative_df)
        #     fn = len(false_negative_df)
        #
        #     if tp + fp == 0:
        #         prec = 0.0
        #     else:
        #         prec = tp / (tp + fp)
        #     if tp + fn == 0:
        #         recall = 0.0
        #     else:
        #         recall = tp / (tp + fn)
        #     if (fp + tn) == 0:
        #         fpr = 0.0
        #     else:
        #         fpr = fp / (fp + tn)
        #     pr.append(
        #         {
        #             'Cutoff': cutoff,
        #             'Precision': prec,
        #             'Recall': recall,
        #             'False Positive Rate': fpr
        #         }
        #     )
        # pr_df = pd.DataFrame(pr)
        # best_fpr_10 = pr_df[pr_df['Recall'] > 0.999999999]['False Positive Rate'].min()
        # best_fpr_99 = pr_df[pr_df['Recall'] > 0.99]['False Positive Rate'].min()
        # best_fpr_95 = pr_df[pr_df['Recall'] > 0.95]['False Positive Rate'].min()
        # self.log('fpr10', best_fpr_10, 4)
        # self.log('fpr95', best_fpr_95, 4)
        # self.log('fpr99', best_fpr_99, 4)
        # self.log('lr', )

        self.test_annotations.clear()
