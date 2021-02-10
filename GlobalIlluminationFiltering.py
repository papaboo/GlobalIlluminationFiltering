import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from Analyze import analyze_dataset
from ImageDataset import ImageDataset
from Metrics import SSIM
import Networks
from Visualize import visualize_result

class GlobalIlluminationFiltering(LightningModule):
    def __init__(self, log_dir=None):
        super().__init__()

        self.loss_function = nn.MSELoss()
        self.net = Networks.ConvNet()
        self.log_dir = log_dir


    def forward(self, input):
        return self.net(input)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    def training_step(self, batch, batch_idx):
        input, target = batch
        inference = self(input)

        # SSIM pr epoch
        with torch.no_grad():
            ssim = SSIM(inference, target).item()
            self.log(f'train_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)

        # Output the first image to see progress
        if batch_idx == 0 and self.log_dir is not None:
            light_tensor = input[0][0,:,:,:]
            albedo_tensor = input[1][0,:,:,:]
            inferred_tensor = inference[0,:,:,:]
            inferred_light_tensor = inferred_tensor / (albedo_tensor + 0.0001)
            reference_tensor = target[0,:,:,:]
            reference_light_tensor = reference_tensor / (albedo_tensor + 0.0001)
            additional_tensors = [["Light", light_tensor],
                                  ["Inferred", inferred_light_tensor],
                                  ["Reference light", reference_light_tensor],
                                  ["Albedo", albedo_tensor]]

            fig = plt.figure(figsize = (25,5))
            fig.suptitle(f"Epoch {self.current_epoch}")
            visualize_result(light_tensor * albedo_tensor, inferred_tensor, reference_tensor, show=False, additional_tensors=additional_tensors)
            plt.savefig(self.log_dir + f"\epoch_{self.current_epoch}.png")
            plt.close(fig)

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
    batch_size=2

    training_set = ImageDataset(["Dataset/classroom/inputs", "Dataset/living-room/inputs", "Dataset/sponza/inputs", "Dataset/sponza-(glossy)/inputs", "Dataset/sponza-(moving-light)/inputs"])
    validation_set = ImageDataset(["Dataset/san-miguel/inputs"])
    validation_data_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=8)

    logger = TensorBoardLogger('tensorboard', name='GlobalIlluminationFiltering')
    log_dir = logger.log_dir

    model = GlobalIlluminationFiltering(log_dir)

    trainer = pl.Trainer(max_epochs=16, gpus=1, profiler="simple", logger=logger)
    trainer.fit(model, DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8), validation_data_loader)

    result = trainer.test(model, validation_data_loader)
    print(result)

    analyze_dataset(model, validation_data_loader, log_dir)