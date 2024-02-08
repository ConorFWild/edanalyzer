import torch
from torch import functional as F
import lightning as lt

from .resnet import resnet18


class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(num_classes=1, num_input=4)

    def forward(self, x):
        return torch.exp(self.resnet(x))

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        x, y = test_batch
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('test_loss', loss)
