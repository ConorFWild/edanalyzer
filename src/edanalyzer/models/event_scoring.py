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


from .resnet import _resnet, BasicBlock, resnet18, resnet10
from .simple_autoencoder import SimpleConvolutionalEncoder, SimpleConvolutionalDecoder
from edanalyzer.losses import categorical_loss


# class Annotation(tables.IsDescription):
#     epoch = tables.Int32Col()
#     idx = tables.Int32Col()
#     f = tables.Int32Col()
#     table = tables.StringCol(32)
#     y = tables.Float32Col()
#     y_hat = tables.Float32Col()
#     set = tables.Int32Col()


annotation_dtype = [
    ('epoch', '<i4'),
    ('idx', '<i4'),
    ('f', '<i4'),
    ('table', '<U32'),
    ('y', '<f4'),
    ('y_hat', '<f4'),
    ('low', '<f4'),
    ('med', '<f4'),
    ('high', '<f4'),
    ('system', '<U32'),
    ('dtag', '<U32'),
    ('event_num', 'i8'),
    ('Confidence', '<U32')
]



class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet10(num_classes=1, num_input=1).float()


# class LitEventScoring(lt.LightningModule):
#     def __init__(self):
#         super().__init__()
#         # self.resnet = resnet10(num_classes=2, num_input=1, headless=True).float()
#         self.resnet = SimpleConvolutionalEncoder()
#
#         self.mol_encoder = SimpleConvolutionalEncoder()
#         self.mol_decoder = SimpleConvolutionalDecoder()
#         self.density_decoder = SimpleConvolutionalDecoder(input_layers=64)
#         # self.fc = nn.Linear(512 + 32, 1)
#         self.fc = nn.Linear(32 + 32, 1)
#
#         self.train_annotations = []
#         self.test_annotations = []
#         self.output = Path('./output/event_scoring_pandda_2')
#
#     def forward(self, x, m, d):
#         mol_encoding = self.mol_encoder(m)
#         density_encoding = self.resnet(x)
#         full_encoding = torch.cat([density_encoding, mol_encoding], dim=1)
#         score = self.fc(full_encoding)
#
#         return F.sigmoid(score)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
#
#     def training_step(self, train_batch, batch_idx):
#         idx, x, m, d, y = train_batch
#         y = y.view(y.size(0), -1)
#         mol_encoding = self.mol_encoder(m)
#         mol_decoding = F.sigmoid( self.mol_decoder(mol_encoding))
#         density_encoding = self.resnet(x)
#         full_encoding = torch.cat([density_encoding, mol_encoding], dim=1)
#         density_decoding = F.sigmoid(self.density_decoder(full_encoding))
#         score = F.sigmoid(self.fc(full_encoding))
#         loss_1 = F.mse_loss(score, y)
#         loss_2 = F.mse_loss(mol_decoding, m)
#         loss_3 = F.mse_loss(density_decoding, d)
#         total_loss = loss_1 + loss_2 + loss_3
#
#         self.log('train_loss', loss_1)
#
#         for j in range(len(idx[0])):
#             self.train_annotations.append(
#                 {
#                     "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
#                     'table': str(idx[0][j]),
#                     "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
#                     "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
#                     'set': 0
#                 }
#             )
#         # self.annotations[]
#         return total_loss
#
#     def validation_step(self, test_batch, batch_idx):
#         idx, x, m, d, y = test_batch
#         y = y.view(y.size(0), -1)
#         mol_encoding = self.mol_encoder(m)
#         density_encoding = self.resnet(x)
#         full_encoding = torch.cat([density_encoding, mol_encoding], dim=1)
#         score = F.sigmoid(self.fc(full_encoding))
#         loss = F.mse_loss(score, y)
#         self.log('test_loss', loss)
#
#         for j in range(len(idx[0])):
#             self.test_annotations.append(
#                 {
#                     "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
#                     'table': str(idx[0][j]),
#                     "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
#                     "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
#                     'set': 1
#                 }
#             )
#
#     def on_train_epoch_end(self):
#         # Log the predictions
#         predictions = self.train_annotations
#         rprint(f"Epoch: {self.trainer.current_epoch}")
#         rprint(predictions)
#         rprint(self.trainer.train_dataloader)
#
#         # Load the table
#         table_file = self.output / 'annotations.h5'
#         if not table_file.exists():
#             fileh = tables.open_file(table_file, mode="w")
#             root = fileh.root
#             fileh.create_table(root, "train_annotations", Annotation)
#             fileh.create_table(root, "test_annotations", Annotation)
#
#         else:
#             fileh = tables.open_file(table_file, mode="a")
#             root = fileh.root
#
#         table = root.train_annotations
#
#         annotation = table.row
#         for _annotation in self.train_annotations:
#             annotation['epoch'] = int(self.trainer.current_epoch)
#             annotation['idx'] = int(_annotation['idx'])
#             annotation['table'] = str(_annotation['table'])
#             annotation['y'] = float(_annotation['y'])
#             annotation['y_hat'] = float(_annotation['y_hat'])
#             annotation['set'] = int(_annotation['set'])
#
#             annotation.append()
#         table.flush()
#         fileh.close()
#
#         self.train_annotations.clear()
#
#     def on_validation_epoch_end(self):
#         # Log the predictions
#         predictions = self.test_annotations
#         rprint(f"Epoch: {self.trainer.current_epoch}")
#         rprint(predictions)
#         # rprint(self.trainer.test_dataloader)
#
#         # Load the table
#         table_file = self.output / 'annotations.h5'
#         if not table_file.exists():
#             fileh = tables.open_file(table_file, mode="w")
#             root = fileh.root
#             fileh.create_table(root, "train_annotations", Annotation)
#             fileh.create_table(root, "test_annotations", Annotation)
#
#         else:
#             fileh = tables.open_file(table_file, mode="a")
#             root = fileh.root
#
#         table = root.test_annotations
#
#         annotation = table.row
#         for _annotation in self.test_annotations:
#             annotation['epoch'] = int(self.trainer.current_epoch)
#             annotation['idx'] = int(_annotation['idx'])
#             annotation['table'] = str(_annotation['table'])
#             annotation['y'] = float(_annotation['y'])
#             annotation['y_hat'] = float(_annotation['y_hat'])
#             annotation['set'] = int(_annotation['set'])
#             annotation.append()
#         table.flush()
#         fileh.close()
#
#         self.test_annotations.clear()


class LitEventScoring(lt.LightningModule):
    def __init__(self, output_dir, config):
        super().__init__()
        # self.automatic_optimization = False
        # self.resnet = resnet10(num_classes=2, num_input=1, headless=True).float()
        # self.z_encoder = SimpleConvolutionalEncoder(input_layers=2)
        self.z_encoder = _resnet(
            'resnet10',
            BasicBlock,
            [config['blocks_1'], config['blocks_2'], config['blocks_3'], config['blocks_4'], ],
            False, False,
            num_classes=3, num_input=2, headless=True,
                                  drop_rate=config['drop_rate'], config=config).float()
        # self.x_encoder = SimpleConvolutionalEncoder(input_layers=1)
        # self.mol_encoder = SimpleConvolutionalEncoder(input_layers=1)
        self.mol_encoder = _resnet(
            'resnet10',
            BasicBlock,
            [config['blocks_1'], config['blocks_2'], config['blocks_3'], config['blocks_4'], ],
            False, False,
            num_classes=2, num_input=1, headless=True,
                                    drop_rate=config['drop_rate'], config=config).float()
        # self.mol_decoder = SimpleConvolutionalDecoder()
        # self.x_decoder = SimpleConvolutionalDecoder(input_layers=512)
        # self.z_decoder = SimpleConvolutionalDecoder(input_layers=512)
        # self.mol_to_weight = nn.Linear(512, 512)
        # self.bn = nn.BatchNorm1d(512)
        # self.fc = nn.Linear(512 + 32, 1)
        self.fc = nn.Sequential(
            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            # nn.Linear(256,128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(config['planes_5']*2, config['combo_layer']),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(16),
            # nn.Linear(config['combo_layer'] ,2),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(512, 256),
            # nn.Dropout(),
            # nn.Linear(16, 2),
            nn.Linear(config['planes_5'] , 3),

        )
        self.train_annotations = []
        self.test_annotations = []

        self.output = output_dir
        self.lr = config['lr']
        self.wd = config['wd']
        self.batch_size = config['batch_size']
        self.ligand = config['ligand']

    def forward(self, x, z, m, d):
        z_encoding = self.z_encoder(z)

        if self.ligand:
            mol_encoding = self.mol_encoder(m)
        else:
            mol_encoding = torch.ones(z_encoding.size()).to('cuda:0')
        # x_encoding = self.x_encoder(x)

        # z_mol_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # z_decoding = F.hardtanh(self.density_decoder(z_mol_encoding), min_val=0.0, max_val=1.0,)
        # mask = torch.zeros(z_decoding.shape).to(z_decoding.device)
        # mask[z_decoding > 0.5] = 1.0
        # full_density = torch.cat(
        #     [
        #         z,
        #         # x * z_decoding
        #         # x
        #         # x * mask
        #     ],
        #     dim=1,
        # )
        # density_encoding = self.density_encoder(full_density)
        # full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # full_encoding = z_encoding * F.hardtanh(self.bn( self.mol_to_weight(mol_encoding)), min_val=-1.0, max_val=1.0)
        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)
        full_encoding = z_encoding * mol_encoding

        score = F.softmax(self.fc(full_encoding))

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, cooldown=10)
        # return [optimizer], [lr_scheduler]
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     # "scheduler": lr_scheduler,
            #     "monitor": "fpr95",
            #     "interval": "epoch",
            #     "frequency": 10,
            #     "strict": False,
            # },
        }
        # return optimizer

    def training_step(self, train_batch, batch_idx):
        idx, x, z, m, d, y = train_batch
        y = y.view(y.size(0), -1)

        # _sample = z[0,0,:,:,:].numpy()
        # _sample_annotation = y[0].numpy()
        # print(
        #     (
        #         f'z\n'
        #         f'Annotation: {_sample_annotation}\n'
        #         f'Mean: {np.mean(_sample)}\n'
        #         f'std: {np.std(_sample)}\n'
        #         f'zeros: {_sample[_sample == 0].size} / {_sample.size}\n'
        #
        #     )
        # )
        #
        # _sample_x = z[0,0,:,:,:].numpy()
        # _sample_x_annotation = y[0].numpy()
        # print(
        #     (
        #         f'x\n'
        #         f'Annotation: {_sample_x_annotation}\n'
        #         f'Mean: {np.mean(_sample_x)}\n'
        #         f'std: {np.std(_sample_x)}\n'
        #         f'zeros: {_sample_x[_sample_x == 0].size} / {_sample_x.size}\n'
        #
        #     )
        # )

        # mol_decoding = F.hardtanh(self.mol_decoder(mol_encoding), min_val=0.0, max_val=1.0,)

        z_encoding = self.z_encoder(z)

        if self.ligand:
            mol_encoding = self.mol_encoder(m)
        else:
            mol_encoding = torch.ones(z_encoding.size()).to('cuda:0')

        # z_mol_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # z_decoding = F.hardtanh(self.z_decoder(z_mol_encoding), min_val=0.0, max_val=1.0)

        # x_encoding = F.sigmoid(self.x_encoder(x))
        # x_mol_encoding = torch.cat([x_encoding, mol_encoding], dim=1)
        # x_decoding = F.hardtanh(self.x_decoder(x_mol_encoding), min_val=0.0, max_val=1.0)



        # mask = torch.zeros(z_decoding.shape).to(z_decoding.device)
        # mask[z_decoding > 0.5] = 1.0
        # full_density = torch.cat(
        #     [
        #         z,
        #         # x * z_decoding
        #         # x,
        #         # x * mask
        #     ],
        #     dim=1,
        # )
        # density_encoding = self.density_encoder(full_density)
        # full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # full_encoding = z_encoding * F.hardtanh(self.bn( self.mol_to_weight(mol_encoding)), min_val=-1.0, max_val=1.0)
        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)
        full_encoding = z_encoding * mol_encoding

        # score = F.sigmoid(self.fc(full_encoding))
        score = F.softmax(self.fc(full_encoding))

        # loss_1 = F.mse_loss(score, y)
        loss_1 = categorical_loss(score, y)
        # loss_2 = F.mse_loss(mol_decoding, m)
        # loss_3 = F.mse_loss(z_decoding, d)
        # loss_4 = F.mse_loss(x_decoding, d)
        total_loss = loss_1 #+ loss_2 + loss_3 + loss_4
        # total_loss = loss_1 * loss_2 * loss_3 * loss_4

        self.log('train_loss', loss_1, batch_size=self.batch_size, )
        # self.log('mol_decode_loss', loss_2)
        # self.log('z_decode_loss', loss_3)
        # self.log('x_decode_loss', loss_4)

        for j in range(len(idx[0])):
            self.train_annotations.append(
                {
                    "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
                    "f": int(idx[2][j].to(torch.device("cpu")).detach().numpy()),
                    'table': str(idx[0][j]),
                    # "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
                    # "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][2],
                    "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][2],
                    "low": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "med": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][1],
                    "high": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][2],
                    'set': 0,
                    "system": str(idx[3][j]),
                    "dtag": str(idx[4][j]),
                    "event_num": int(idx[5][j]),
                    'Confidence': str(idx[-1][j])
                }
            )
        # self.annotations[]
        return total_loss

    def validation_step(self, test_batch, batch_idx):
        idx, x, z, m, d, y = test_batch
        y = y.view(y.size(0), -1)

        # print(f'Mol Sum Density: {torch.sum(m[0,:,:,:])}')

        z_encoding = self.z_encoder(z)

        if self.ligand:
            mol_encoding = self.mol_encoder(m)
        else:
            mol_encoding = torch.ones(z_encoding.size()).to('cuda:0')
        # x_encoding = F.sigmoid( self.x_encoder(x))
        # z_mol_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # z_decoding = F.hardtanh(self.density_decoder(z_mol_encoding), min_val=0.0, max_val=1.0,)
        # mask = torch.zeros(z_decoding.shape).to(z_decoding.device)
        # mask[z_decoding > 0.5] = 1.0
        # full_density = torch.cat(
        #     [
        #         z,
        #         # x * z_decoding
        #         # x,
        #         # x * mask
        #     ],
        #     dim=1,
        # )
        # full_density = z
        # density_encoding = self.density_encoder(full_density)
        # full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        # full_encoding =  z_encoding * F.hardtanh(self.bn( self.mol_to_weight(mol_encoding)), min_val=-1.0, max_val=1.0)
        # full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)
        full_encoding = z_encoding * mol_encoding

        # print(f'Z Encoding: {z_encoding[0,:10]}')
        # print(f'Mol Encoding: {mol_encoding[0,:10]}')

        # score = F.sigmoid(self.fc(full_encoding))
        score = F.softmax(self.fc(full_encoding))
        # print(f'Score: {score[0,:]}')
        # print('#####################################')


        loss = categorical_loss(score, y)
        # loss = F.mse_loss(score, y)
        self.log('test_loss', loss, batch_size=12)

        for j in range(len(idx[0])):
            self.test_annotations.append(
                {
                    "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
                    "f": int(idx[2][j].to(torch.device("cpu")).detach().numpy()),
                    'table': str(idx[0][j]),
                    # "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
                    # "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][2],
                    "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][2],
                    "low": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "med": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][1],
                    "high": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][2],
                    'set': 1,
                    "system": str(idx[3][j]),
                    "dtag": str(idx[4][j]),
                    "event_num": int(idx[5][j]),
                    'Confidence': str(idx[-1][j])
                }
            )

    def on_train_epoch_end(self):
        # Log the predictions
        predictions = self.train_annotations
        # rprint(f"Epoch: {self.trainer.current_epoch}")
        # rprint(predictions)
        # rprint(self.trainer.train_dataloader)

        # Load the table
        # table_file = self.output / 'annotations.h5'
        # if not table_file.exists():
        #     fileh = tables.open_file(table_file, mode="w")
        #     root = fileh.root
        #     fileh.create_table(root, "train_annotations", Annotation)
        #     fileh.create_table(root, "test_annotations", Annotation)
        #
        # else:
        #     fileh = tables.open_file(table_file, mode="a")
        #     root = fileh.root
        #
        #
        #
        # table = root.train_annotations
        #
        # annotation = table.row
        # for _annotation in self.train_annotations:
        #     annotation['epoch'] = int(self.trainer.current_epoch)
        #     annotation['idx'] = int(_annotation['idx'])
        #     annotation['f'] = int(_annotation['f'])
        #
        #     annotation['table'] = str(_annotation['table'])
        #     annotation['y'] = float(_annotation['y'])
        #     annotation['y_hat'] = float(_annotation['y_hat'])
        #     annotation['set'] = int(_annotation['set'])
        #
        #     annotation.append()
        # table.flush()
        # fileh.close()

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
                        int(_annotation['idx']),
                        int(_annotation['f']),
                        str(_annotation['table']),
                        float(_annotation['y']),
                        float(_annotation['y_hat']),
                        float(_annotation['low']),
                        float(_annotation['med']),
                        float(_annotation['high']),
                        str(_annotation['system']),
                        str(_annotation['dtag']),
                        int(_annotation['event_num']),
                        str(_annotation['Confidence'])
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
        # rprint(f"Epoch: {self.trainer.current_epoch}")
        # rprint(predictions)
        # rprint(self.trainer.test_dataloader)

        ## Load the table
        # table_file = self.output / 'annotations.h5'
        # if not table_file.exists():
        #     fileh = tables.open_file(table_file, mode="w")
        #     root = fileh.root
        #     fileh.create_table(root, "train_annotations", Annotation)
        #     fileh.create_table(root, "test_annotations", Annotation)
        #
        # else:
        #     fileh = tables.open_file(table_file, mode="a")
        #     root = fileh.root
        #
        # table = root.test_annotations
        #
        # annotation = table.row
        # for _annotation in self.test_annotations:
        #     annotation['epoch'] = int(self.trainer.current_epoch)
        #     annotation['idx'] = int(_annotation['idx'])
        #     annotation['f'] = int(_annotation['f'])
        #     annotation['table'] = str(_annotation['table'])
        #     annotation['y'] = float(_annotation['y'])
        #     annotation['y_hat'] = float(_annotation['y_hat'])
        #     annotation['set'] = int(_annotation['set'])
        #     annotation.append()
        # table.flush()
        # fileh.close()

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
                        int(_annotation['idx']),
                        int(_annotation['f']),
                        str(_annotation['table']),
                        float(_annotation['y']),
                        float(_annotation['y_hat']),
                        float(_annotation['low']),
                        float(_annotation['med']),
                        float(_annotation['high']),
                        str(_annotation['system']),
                        str(_annotation['dtag']),
                        int(_annotation['event_num']),
                        str(_annotation['Confidence'])

        )
                    for _annotation in self.test_annotations
                ],
                dtype=annotation_dtype
            )
        test_annotation_table.append(annotations)

        #
        # _epoch = epoch
        # best_df = df[(df['epoch'] == _epoch)]
        best_df = pd.DataFrame.from_records(annotations)
        pr = []



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

        best_fpr_95 = get_fpr(best_df, 0.05)
        best_fpr_99 = get_fpr(best_df, 0.01)
        best_fpr_10 = get_fpr(best_df, 0.0)

        self.log('fpr10', best_fpr_10, 4, sync_dist=True)
        self.log('fpr95', best_fpr_95, 4, sync_dist=True)
        self.log('fpr99', best_fpr_99, 4, sync_dist=True)
        # self.log('lr', )

        fpr99s = []
        for _system in best_df['system'].unique():
            fpr99s.append(get_fpr(best_df[best_df['system'] == _system], 0.01))


        self.log('medianfpr99', np.median(fpr99s), 4, sync_dist=True)

        df = pd.DataFrame(
            [
                {
                    'y': float(_annotation['y']),
                    'y_hat': float(_annotation['y_hat']),
                    'system': str(_annotation['system']),
                    'dtag': str(_annotation['dtag']),
                    'event_num': int(_annotation['event_num']),
                }

        for _annotation in self.test_annotations
        ]
        )

        # print('df')
        # print(df)
        dataset_max_ys = df.groupby(['dtag', ])['y'].transform('max')
        # print('dataset_max_ys')
        # print(dataset_max_ys)
        dataset_num_ys = df.groupby(['dtag', ])['y'].transform('nunique')
        # print('dataset_num_ys')
        # print(dataset_num_ys)
        multi_event_datasets = df[(dataset_max_ys == 1.0) & (dataset_num_ys > 1)]
        # print('multi_event_datasets')
        # print(multi_event_datasets)
        dataset_best = multi_event_datasets[
            multi_event_datasets['y_hat'] == multi_event_datasets.groupby('dtag')['y_hat'].transform('max')]

        if len(dataset_best) == 0:
            best_scorer_hit = 0.0
        else:
            best_scorer_hit = len(dataset_best[dataset_best['y'] == 1.0]) / len(dataset_best)
        self.log('best_scorer_hit', best_scorer_hit, 4, sync_dist=True)

        self.test_annotations.clear()

        # sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        # if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     sch.step(self.trainer.callback_metrics["test_loss"])

    # def on_after_backward(self):
    #     print(f'UNUSUED PARAMS:')
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)


def get_fpr(best_df, recall):
    # true_hit_best_df = best_df[best_df['y'] > 0.9]
    # negative_best_df = best_df[best_df['y'] <= 0.1]
    true_hit_best_df = best_df[best_df['Confidence'] == 'High']
    negative_best_df = best_df[best_df['Confidence'] == 'Low']

    cutoff = true_hit_best_df['y_hat'].quantile(recall)

    false_hit_df = negative_best_df[negative_best_df['y_hat'] >= cutoff]
    true_negative_df = negative_best_df[negative_best_df['y_hat'] < cutoff]
    fp = len(false_hit_df)
    tn = len(true_negative_df)

    if (fp + tn) == 0:
        _fpr = 0.0
    else:
        _fpr = fp / (fp + tn)

    return _fpr
