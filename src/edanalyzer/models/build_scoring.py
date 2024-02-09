from rich import print as rprint
import torch
from torch.nn import functional as F
import lightning as lt

from .resnet import resnet18


class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(num_classes=1, num_input=4).float()
        self.annotations = []

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
                "idx": idx[j],
                "y": y[j],
                "y_hat": score[j]
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
        predictions = self.training_step_outputs
        rprint(predictions)
        rprint(self.trainer.train_dataloader)
        self.annotations.clear()