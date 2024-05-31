import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

os.environ["HF_HOME"] = "/pscratch/sd/a/archis/huggingface"

from sentence_transformers import SentenceTransformer
import numpy as np
from torch import optim, nn
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from matplotlib import pyplot as plt


class SpectrumDecoder(nn.Module):
    def __init__(self, model_params):
        super(SpectrumDecoder, self).__init__()
        self.nx = model_params["nx"]
        self.ny = model_params["ny"]
        num_channels = model_params["num_channels"]

        self.fc = nn.Linear(model_params["emb_size"], self.nx * self.ny)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")

        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

        self.conv3_L = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
        self.conv4_L = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

    def forward(self, embs_sum):
        transformed_embs = F.relu(self.fc(embs_sum.to("cuda")))
        transformed_embs = transformed_embs.reshape((-1, 1, self.nx, self.ny))
        transformed_embs = F.relu(self.conv1(transformed_embs))
        transformed_embs = F.relu(self.conv2(transformed_embs))

        embs_H = F.relu(self.conv3(transformed_embs))
        spectrum_H = F.relu(self.conv4(embs_H))

        embs_L = F.relu(self.conv3_L(transformed_embs))
        spectrum_L = F.relu(self.conv4_L(embs_L))

        return spectrum_H, spectrum_L


class ZEUSElectronSpectrumator(nn.Module):
    def __init__(self, inputs, outputs, batch_size, model_params):
        super(ZEUSElectronSpectrumator, self).__init__()
        self.vit_encoder = SentenceTransformer("clip-ViT-L-14")
        self.vit_encoder.requires_grad_(False)
        self.decoder = SpectrumDecoder(model_params=model_params)

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
    def __init__(self, learning_rate, batch_size, model_params, log_dir):
        super().__init__()
        self.learning_rate = learning_rate
        self.inputs = ["Pointing", "Interf"]
        self.outputs = ["EspecH-downsampled", "EspecL-downsampled"]
        self.zeus = ZEUSElectronSpectrumator(self.inputs, self.outputs, batch_size, model_params).to("cuda")
        self.espech_preds = []
        self.especl_preds = []
        self.espech_actuals = []
        self.especl_actuals = []
        self.val_batch_idxs = []
        self.mse_h = MeanSquaredError()
        self.mse_l = MeanSquaredError()
        self.log_dir = log_dir

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

        self.log("espech-loss", espech_loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("especl-loss", especl_loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        espech_loss, especl_loss, espech_hat, especl_hat = self.step(batch)
        loss = espech_loss + especl_loss

        self.log("val-espech-loss", espech_loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("val-especl-loss", especl_loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("val-loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        self.espech_preds.append(espech_hat)
        self.especl_preds.append(especl_hat)
        self.espech_actuals.append(batch["EspecH-downsampled"])
        self.especl_actuals.append(batch["EspecL-downsampled"])
        self.val_batch_idxs.append(batch_idx)

        return loss

    def predict_step(self, batch, batch_idx):
        espech_hat, especl_hat = self(batch)
        return {
            "espech-actual": batch["EspecH-downsampled"],
            "especl-actual": batch["EspecL-downsampled"],
            "prompt": batch["prompt"],
            "batch_idx": batch_idx,
            "espech": espech_hat,
            "especl": especl_hat,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        os.makedirs(os.path.join(self.output_dir, "binary"), exist_ok=True)
        torch.save(predictions, os.path.join(self.output_dir, "binary", f"predictions_{trainer.global_rank}.pt"))
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))

        os.makedirs(os.path.join(self.output_dir, "plots", f"final"), exist_ok=True)
        fig = plt.figure(figsize=(10, 4), tight_layout=True)
        for batch, batch_idx in zip(predictions, batch_indices):
            for nm, preds, actuals, bidx in zip(
                ["espech", "especl"],
                [batch["espech"], batch["especl"]],
                [batch["espech-actual"], batch["especl-actual"]],
                [batch_idx, batch_idx],
            ):
                os.makedirs(figdir := os.path.join(self.output_dir, "plots", "final", nm), exist_ok=True)

                for j, (_bidx, pred, actual) in enumerate(zip(bidx, preds, actuals)):
                    figpath = os.path.join(figdir, f"{nm}-{_bidx}.png")
                    ax = fig.add_subplot(1, 2, 1)
                    cb = ax.contourf(pred.cpu().detach().numpy())
                    fig.colorbar(cb)

                    ax = fig.add_subplot(1, 2, 2)
                    cb = ax.contourf(actual.cpu().detach().numpy())
                    fig.colorbar(cb)

                    fig.savefig(figpath, bbox_inches="tight")
                    fig.clf()
        plt.close()
