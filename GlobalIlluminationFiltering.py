import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from Analyze import analyze_dataset
from ImageDataset import ImageDataset
from Visualize import visualize_result
from Metrics import SSIM

class GlobalIlluminationFiltering(LightningModule):
    def __init__(self):
        super().__init__()

        self.loss_function = lambda x, y: ((x - y) ** 2).mean()

        self.features = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.CELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.CELU(inplace=True)
        )


    def forward(self, input):
        color, albedo, normals, positions = input

        # Factor out albedo
        light = color / (albedo + 0.00001) # Slight bias to avoid division by zero

        x = torch.cat((light, albedo, normals), dim=1)
        filtered_light = self.features(x)

        # Multiply by albedo
        filtered_color = filtered_light * albedo

        # TODO Bilateral filter based on infered depth and std dev

        return filtered_color


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    def training_step(self, batch, batch_idx):
        input, target = batch
        inference = self(input)

        # SSIM pr epoch
        with torch.no_grad():
            ssim = SSIM(inference, target).item()
            self.log(f'train_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)

        return self.loss_function(inference, target)


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    # Print a newline when epoch ends to keep the progress bar from the previous epoch
    # Unfortunately per training epoch logging is done after this, so those results are shown in the next epochs progress bar.
    def training_epoch_end(self, foo):
        print('\n')


    def evaluation_step(self, batch, batch_idx, step_name):
        input, target = batch
        inference = self(input)

        # Loss
        loss = self.loss_function(inference, target).item()
        self.log(f'{step_name}_loss', loss, on_epoch=True, prog_bar=True)

        # SSIM
        ssim = SSIM(inference, target).item()
        self.log(f'{step_name}_ssim', ssim, on_epoch=True, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, "val")


    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, "test")


if __name__ == '__main__':
    partial_set = False

    training_set = ImageDataset(["Dataset/classroom/inputs", "Dataset/living-room/inputs", "Dataset/sponza/inputs", "Dataset/sponza-(glossy)/inputs", "Dataset/sponza-(moving-light)/inputs"], partial_set=partial_set)
    validation_set = ImageDataset(["Dataset/san-miguel/inputs"], partial_set=partial_set)
    validation_data_loader = DataLoader(validation_set, batch_size=8, num_workers=8)

    model = GlobalIlluminationFiltering()

    logger = TensorBoardLogger('tensorboard', name='GlobalIlluminationFiltering')
    log_dir = logger.log_dir
    trainer = pl.Trainer(max_epochs=1, gpus=1, profiler="simple", logger=logger)
    trainer.fit(model, DataLoader(training_set, batch_size=8, shuffle=True, num_workers=8), validation_data_loader)

    result = trainer.test(model, validation_data_loader)
    print(result)

    analyze_dataset(model, validation_data_loader, log_dir)