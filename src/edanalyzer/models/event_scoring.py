from pathlib import Path

from rich import print as rprint
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt
import yaml
import tables

from .resnet import resnet18, resnet10
from .simple_autoencoder import SimpleConvolutionalEncoder, SimpleConvolutionalDecoder
from edanalyzer.losses import categorical_loss


class Annotation(tables.IsDescription):
    epoch = tables.Int32Col()
    idx = tables.Int32Col()
    table = tables.StringCol(32)
    y = tables.Float32Col()
    y_hat = tables.Float32Col()
    set = tables.Int32Col()


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
    def __init__(self):
        super().__init__()
        # self.resnet = resnet10(num_classes=2, num_input=1, headless=True).float()
        self.resnet = SimpleConvolutionalEncoder()
        self.density_encoder = SimpleConvolutionalEncoder(input_layers=2)
        self.mol_encoder = SimpleConvolutionalEncoder()
        self.mol_decoder = SimpleConvolutionalDecoder()
        self.density_decoder = SimpleConvolutionalDecoder(input_layers=256)
        # self.fc = nn.Linear(512 + 32, 1)
        self.fc = nn.Linear(256, 2)

        self.train_annotations = []
        self.test_annotations = []
        self.output = Path('./output/event_scoring_cat')

    def forward(self, x, z, m, d):
        mol_encoding = self.mol_encoder(m)
        z_encoding = self.resnet(z)
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
        full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)

        score = F.sigmoid(self.fc(full_encoding))

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        idx, x, z, m, d, y = train_batch
        y = y.view(y.size(0), -1)

        mol_encoding = self.mol_encoder(m)
        mol_decoding = F.hardtanh(self.mol_decoder(mol_encoding), min_val=0.0, max_val=1.0,)
        z_encoding = self.resnet(z)
        z_mol_encoding = torch.cat([z_encoding, mol_encoding], dim=1)
        z_decoding = F.hardtanh(self.density_decoder(z_mol_encoding), min_val=0.0, max_val=1.0,)
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
        full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)

        # score = F.sigmoid(self.fc(full_encoding))
        score = F.softmax(self.fc(full_encoding))

        # loss_1 = F.mse_loss(score, y)
        loss_1 = categorical_loss(score, y)
        loss_2 = F.mse_loss(mol_decoding, m)
        loss_3 = F.mse_loss(z_decoding, d)
        total_loss = loss_1 + loss_2 + loss_3

        self.log('train_loss', loss_1)
        self.log('mol_decode_loss', loss_2)
        self.log('z_decode_loss', loss_3)

        for j in range(len(idx[0])):
            self.train_annotations.append(
                {
                    "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
                    'table': str(idx[0][j]),
                    # "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
                    # "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][1],
                    "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][1],
                    'set': 0
                }
            )
        # self.annotations[]
        return total_loss

    def validation_step(self, test_batch, batch_idx):
        idx, x, z, m, d, y = test_batch
        y = y.view(y.size(0), -1)

        mol_encoding = self.mol_encoder(m)
        z_encoding = self.resnet(z)
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
        full_encoding = torch.cat([z_encoding, mol_encoding], dim=1)

        # score = F.sigmoid(self.fc(full_encoding))
        score = F.softmax(self.fc(full_encoding))

        loss = categorical_loss(score, y)
        # loss = F.mse_loss(score, y)
        self.log('test_loss', loss)

        for j in range(len(idx[0])):
            self.test_annotations.append(
                {
                    "idx": int(idx[1][j].to(torch.device("cpu")).detach().numpy()),
                    'table': str(idx[0][j]),
                    # "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][0],
                    # "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][0],
                    "y": [float(x) for x in y[j].to(torch.device("cpu")).detach().numpy()][1],
                    "y_hat": [float(x) for x in score[j].to(torch.device("cpu")).detach().numpy()][1],
                    'set': 1
                }
            )

    def on_train_epoch_end(self):
        # Log the predictions
        predictions = self.train_annotations
        rprint(f"Epoch: {self.trainer.current_epoch}")
        rprint(predictions)
        rprint(self.trainer.train_dataloader)

        # Load the table
        table_file = self.output / 'annotations.h5'
        if not table_file.exists():
            fileh = tables.open_file(table_file, mode="w")
            root = fileh.root
            fileh.create_table(root, "train_annotations", Annotation)
            fileh.create_table(root, "test_annotations", Annotation)

        else:
            fileh = tables.open_file(table_file, mode="a")
            root = fileh.root

        table = root.train_annotations

        annotation = table.row
        for _annotation in self.train_annotations:
            annotation['epoch'] = int(self.trainer.current_epoch)
            annotation['idx'] = int(_annotation['idx'])
            annotation['table'] = str(_annotation['table'])
            annotation['y'] = float(_annotation['y'])
            annotation['y_hat'] = float(_annotation['y_hat'])
            annotation['set'] = int(_annotation['set'])

            annotation.append()
        table.flush()
        fileh.close()

        self.train_annotations.clear()

    def on_validation_epoch_end(self):
        # Log the predictions
        predictions = self.test_annotations
        rprint(f"Epoch: {self.trainer.current_epoch}")
        rprint(predictions)
        # rprint(self.trainer.test_dataloader)

        # Load the table
        table_file = self.output / 'annotations.h5'
        if not table_file.exists():
            fileh = tables.open_file(table_file, mode="w")
            root = fileh.root
            fileh.create_table(root, "train_annotations", Annotation)
            fileh.create_table(root, "test_annotations", Annotation)

        else:
            fileh = tables.open_file(table_file, mode="a")
            root = fileh.root

        table = root.test_annotations

        annotation = table.row
        for _annotation in self.test_annotations:
            annotation['epoch'] = int(self.trainer.current_epoch)
            annotation['idx'] = int(_annotation['idx'])
            annotation['table'] = str(_annotation['table'])
            annotation['y'] = float(_annotation['y'])
            annotation['y_hat'] = float(_annotation['y_hat'])
            annotation['set'] = int(_annotation['set'])
            annotation.append()
        table.flush()
        fileh.close()

        self.test_annotations.clear()
