import os, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

os.environ["HF_HOME"] = "/pscratch/sd/a/archis/huggingface"

from sentence_transformers import SentenceTransformer, util

import os
from torch import optim, nn
import lightning as L
from matplotlib import pyplot as plt


class SpectrumDecoder(nn.Module):
    def __init__(self):
        super(SpectrumDecoder, self).__init__()
        self.fc = nn.Linear(2304, 82 * 76)
        num_channels = 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding="same")
        # Layer 2: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
        # Layer 3: Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
        # Layer 4: Convolutional Layer
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

        self.conv3_L = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
        # Layer 4: Convolutional Layer
        self.conv4_L = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

    def forward(self, embs_sum):
        transformed_embs = F.relu(self.fc(embs_sum.to("cuda")))
        transformed_embs = transformed_embs.reshape((-1, 1, 82, 76))
        transformed_embs = F.relu(self.conv1(transformed_embs))
        transformed_embs = F.relu(self.conv2(transformed_embs))

        embs_H = F.relu(self.conv3(transformed_embs))
        spectrum_H = F.relu(self.conv4(embs_H))

        embs_L = F.relu(self.conv3_L(transformed_embs))
        spectrum_L = F.relu(self.conv4_L(embs_L))

        return spectrum_H, spectrum_L


class ZEUSElectronSpectrumator(nn.Module):
    def __init__(self, inputs, outputs, batch_size):
        super(ZEUSElectronSpectrumator, self).__init__()
        self.vit_encoder = SentenceTransformer("clip-ViT-L-14")
        self.vit_encoder.requires_grad_(False)
        self.decoder = SpectrumDecoder()

        self.input_names = inputs
        self.output_names = outputs
        self.batch_size = batch_size

    def forward(self, input_dict):
        prompt_embs = [
            (torch.sum(torch.as_tensor(self.vit_encoder.encode(sample)), axis=0)[None, :])
            for sample in input_dict["prompt"]
        ]
        prompt_embs = torch.cat(prompt_embs)
        emb_2d = torch.cat([torch.as_tensor(self.vit_encoder.encode(input_dict[k])) for k in self.input_names], axis=-1)
        embs = torch.cat([prompt_embs, emb_2d], axis=-1)

        spectrum_H, spectrum_L = self.decoder(embs)

        return torch.squeeze(spectrum_H, dim=1), torch.squeeze(spectrum_L, dim=1)


# define the LightningModule
class ZEUSLightningModule(L.LightningModule):
    def __init__(self, learning_rate, batch_size):
        super().__init__()
        self.learning_rate = learning_rate
        self.inputs = ["Pointing", "Interf"]
        self.outputs = ["EspecH-downsampled", "EspecL-downsampled"]
        self.zeus = ZEUSElectronSpectrumator(self.inputs, self.outputs, batch_size).to("cuda")
        self.espech_preds = []  # torch.zeros((batch_size, 82, 76))
        self.especl_preds = []  # torch.zeros((batch_size, 82, 76))
        self.espech_actuals = []
        self.especl_actuals = []
        self.val_batch_idxs = []
        self.mse_h = MeanSquaredError()
        self.mse_l = MeanSquaredError()

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in self.zeus.input_names + ["prompt"]}
        espech_hat, especl_hat = self.zeus(inputs)

        return espech_hat, especl_hat

    def step(self, batch):
        espech_hat, especl_hat = self(batch)

        espech_loss = self.mse_h(espech_hat, torch.stack(batch["EspecH-downsampled"]).to("cuda"))
        especl_loss = self.mse_l(especl_hat, torch.stack(batch["EspecL-downsampled"]).to("cuda"))
        return espech_loss, especl_loss, espech_hat, especl_hat

    def training_step(self, batch, batch_idx):
        espech_loss, especl_loss, _, _ = self.step(batch)
        loss = espech_loss + especl_loss

        self.log("loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        espech_loss, especl_loss, espech_hat, especl_hat = self.step(batch)
        loss = espech_loss + especl_loss

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.espech_preds.append(espech_hat)
        self.especl_preds.append(especl_hat)
        self.espech_actuals.append(batch["EspecH-downsampled"])
        self.especl_actuals.append(batch["EspecL-downsampled"])
        self.val_batch_idxs.append(batch_idx)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_validation_epoch_end(self):
        if self.current_epoch % 38 == 0:
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "plots"))
                os.makedirs(figdir := os.path.join(td, "plots", f"epoch-{self.current_epoch}"))
                fig = plt.figure(figsize=(10, 4), tight_layout=True)
                for i in range(len(self.espech_preds)):
                    for j in range(len(self.espech_preds[i])):
                        figpath = os.path.join(figdir, f"batch={self.val_batch_idxs[i]}-espech-{j}.png")
                        ax = fig.add_subplot(1, 2, 1)
                        cb = ax.contourf(self.espech_preds[i][j].cpu().detach().numpy())
                        fig.colorbar(cb)

                        ax = fig.add_subplot(1, 2, 2)
                        cb = ax.contourf(self.espech_actuals[i][j].cpu().detach().numpy())
                        fig.colorbar(cb)

                        fig.savefig(figpath, bbox_inches="tight")
                        fig.clf()
                self.logger.experiment.log_artifacts(self.logger.run_id, td)
                plt.close()
