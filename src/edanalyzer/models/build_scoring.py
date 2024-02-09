import torch
from torch.nn import functional as F
import lightning as lt

from .resnet import resnet18


class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(num_classes=1, num_input=4).float()

    def forward(self, x):

        return torch.exp(self.resnet(x))

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.view(y.size(0), -1)
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.view(y.size(0), -1)
        score = torch.exp(self.resnet(x))
        loss = F.mse_loss(score, y)
        self.log('test_loss', loss)
