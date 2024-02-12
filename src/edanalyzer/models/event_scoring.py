from pathlib import Path

from rich import print as rprint
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt
import yaml
import tables

from .resnet import resnet18

class Annotation(tables.IsDescription):
    epoch = tables.Int32Col()
    idx = tables.Int32Col()
    y = tables.Float32Col()
    y_hat = tables.Float32Col()


class LitEventScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(num_classes=1, num_input=2).float()
        self.annotations = []
        self.output = Path('./output/event_scoring')

    def forward(self, x):
        return F.softmax(self.resnet(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        idx, x, y = train_batch
        # y = y.view(y.size(0),  -1)
        score = F.softmax(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('train_loss', loss)

        for j in range(idx.size(0)):

            self.annotations.append(
                    {
                    "idx": int(idx[j].to(torch.device("cpu")).detach().numpy()),
                    "y": float(y[j].to(torch.device("cpu")).detach().numpy()[0]),
                    "y_hat": float(score[j].to(torch.device("cpu")).detach().numpy())
                }
            )
        # self.annotations[]
        return loss

    def validation_step(self, test_batch, batch_idx):
        idx, x, y = test_batch
        # y = y.view(y.size(0), -1)
        score = F.softmax(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('test_loss', loss)

        for j in range(idx.size(0)):

            self.annotations.append(
                    {
                        "idx": int(idx[j].to(torch.device("cpu")).detach().numpy()),
                        "y": float(y[j].to(torch.device("cpu")).detach().numpy()[0]),
                        "y_hat": float(score[j].to(torch.device("cpu")).detach().numpy())
                    }
            )

    def on_train_epoch_end(self):
        # Log the predictions
        predictions = self.annotations
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
        for _annotation in self.annotations:
            annotation['epoch'] = int(self.trainer.current_epoch)
            annotation['idx'] = int(_annotation['idx'])
            annotation['y'] = float(_annotation['y'])
            annotation['y_hat'] = float(_annotation['y_hat'])
            annotation.append()
        table.flush()
        fileh.close()

        self.annotations.clear()

    def on_validation_epoch_end(self):
        # Log the predictions
        predictions = self.annotations
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

        table = root.test_annotations

        annotation = table.row
        for _annotation in self.annotations:
            annotation['epoch'] = int(self.trainer.current_epoch)
            annotation['idx'] = int(_annotation['idx'])
            annotation['y'] = float(_annotation['y'])
            annotation['y_hat'] = float(_annotation['y_hat'])
            annotation.append()
        table.flush()
        fileh.close()

        self.annotations.clear()