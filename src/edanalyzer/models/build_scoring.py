from pathlib import Path

from rich import print as rprint
import torch
from torch.nn import functional as F
import lightning as lt
import yaml

from .resnet import resnet18


class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(num_classes=1, num_input=4).float()
        self.annotations = []
        self.output = Path('./output/build_scoring')

    def forward(self, x):

        return torch.exp(self.resnet(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        idx, x, y = train_batch
        y = y.view(y.size(0), -1)
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('train_loss', loss)

        self.annotations.append(
            [
                {
                "idx": int(idx[j].to(torch.device("cpu")).detach().numpy()),
                "y": float(y[j].to(torch.device("cpu")).detach().numpy()[0]),
                "y_hat": float(score[j].to(torch.device("cpu")).detach().numpy())
            }
                for j in range(idx.size(0))
            ]
        )
        # self.annotations[]
        return loss

    def validation_step(self, test_batch, batch_idx):
        idx, x, y = test_batch
        y = y.view(y.size(0), -1)
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('test_loss', loss)

    def on_train_epoch_end(self):
        # Log the predictions
        # predictions = self.training_step_outputs
        predictions = self.annotations
        rprint(f"Epoch: {self.trainer.current_epoch}")
        rprint(predictions)
        rprint(self.trainer.train_dataloader)

        if not (self.output / 'annotations_train.yaml').exists():
            annotations = {}

        else:
            with open(self.output / 'annotations_train.yaml', 'r') as f:
                annotations = yaml.safe_load(f)

        annotations[self.trainer.current_epoch] = self.annotations

        with open(self.output / 'annotations_train.yaml', 'w') as f:
            yaml.dump(annotations, f)

        self.annotations.clear()
