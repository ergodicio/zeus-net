import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

os.environ["HF_HOME"] = "/pscratch/sd/a/archis/huggingface"

from sentence_transformers import SentenceTransformer
from torch import optim, nn
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from matplotlib import pyplot as plt

from misc import export_run


class SpectrumDecoder(nn.Module):
    def __init__(self, model_params):
        super(SpectrumDecoder, self).__init__()
        self.nx = model_params["nx"]
        self.ny = model_params["ny"]
        num_channels = model_params["num_channels"]

        self.fc = nn.Linear(8256, self.nx * self.ny)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")

        self.h_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
                for _ in range(model_params["num_layers"])
            ]
        )
        self.h_out = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

        self.l_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding="same")
                for _ in range(model_params["num_layers"])
            ]
        )
        self.l_out = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding="same")

    def forward(self, embs_sum):
        transformed_embs = F.relu(self.fc(embs_sum.to("cuda")))
        transformed_embs = transformed_embs.reshape((-1, 1, self.nx, self.ny))
        transformed_embs = F.relu(self.conv1(transformed_embs))
        transformed_embs = F.relu(self.conv2(transformed_embs))

        embs_H = F.relu(self.h_convs[0](transformed_embs))
        for conv in self.h_convs[1:]:
            embs_H = F.relu(conv(embs_H))

        spectrum_H = F.relu(self.h_out(embs_H))

        embs_L = F.relu(self.l_convs[0](transformed_embs))
        for conv in self.l_convs[1:]:
            embs_L = F.relu(conv(embs_L))

        spectrum_L = F.relu(self.l_out(embs_L))

        return spectrum_H, spectrum_L


class ZEUSElectronSpectrumator(nn.Module):
    def __init__(self, inputs, outputs, batch_size, model_params):
        super(ZEUSElectronSpectrumator, self).__init__()
        self.vit_encoder = SentenceTransformer("clip-ViT-L-14")
        self.vit_encoder.requires_grad_(False)
        self.decoder = SpectrumDecoder(model_params=model_params)

        self.prompt_nns = nn.ModuleList([nn.Linear(768, 192) for _ in range(35)]).to("cuda")

        self.input_names = inputs
        self.output_names = outputs
        self.batch_size = batch_size

    def forward(self, input_dict):

        # the vit encoder produces a 768-dimensional embedding for each input
        # we send these embeddings through a learned layer and downsize them to 192 and then concatenate them
        # into a 32 x 6720 object

        prompt_embs_of_one_batch = []
        for sample in input_dict["prompt"]:
            prompt_embs = torch.zeros((35, 192)).to("cuda")
            prompt_embs_of_one_sample = torch.as_tensor(self.vit_encoder.encode(sample)).to("cuda")  # 35 x 768
            for pid, pnn in enumerate(self.prompt_nns):
                prompt_embs[pid] = pnn(prompt_embs_of_one_sample[pid])  # 35 x 192

            prompt_embs_of_one_batch.append(prompt_embs)

        prompt_embs = (
            torch.stack(prompt_embs_of_one_batch).to("cuda").reshape(len(input_dict["prompt"]), -1)
        )  # batch_size x 35 x 192

        # the 2d embeddings are also 768-dimensional
        emb_2d = torch.cat(
            [torch.as_tensor(self.vit_encoder.encode(input_dict[k])) for k in self.input_names], axis=-1
        ).to("cuda")

        # the embeddings are concatenated along the last axis so the total number is
        # 6720 + 2*768 = 8256
        # the 8256 is ideally enough information to reproduce the spectra

        embs = torch.cat([prompt_embs, emb_2d], axis=-1)
        spectrum_H, spectrum_L = self.decoder(embs)

        return torch.squeeze(spectrum_H, dim=1), torch.squeeze(spectrum_L, dim=1)


# define the LightningModule
class ZEUSLightningModule(L.LightningModule):
    def __init__(self, learning_rate, batch_size, model_params, run_id, log_dir):
        super().__init__()
        self.save_hyperparameters()
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
        self.run_id = run_id
        self.log_loss = model_params["log_loss"]

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in self.zeus.input_names + ["prompt"]}
        espech_hat, especl_hat = self.zeus(inputs)
        if self.log_loss:
            espech_hat = torch.exp(torch.abs(espech_hat))
            especl_hat = torch.exp(torch.abs(especl_hat))

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

    def on_predict_end(self):
        if self.trainer.is_global_zero:
            self.logger.experiment.log_artifacts(self.run_id, self.log_dir)
            export_run(self.run_id)
        else:
            time.sleep(100)

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
        for batch, samps_in_batch in zip(predictions, batch_indices[0]):
            for nm, preds, actuals, _samps_in_batch in zip(
                ["espech", "especl"],
                [batch["espech"], batch["especl"]],
                [batch["espech-actual"], batch["especl-actual"]],
                [samps_in_batch, samps_in_batch],
            ):
                os.makedirs(figdir := os.path.join(self.output_dir, "plots", "final", nm), exist_ok=True)

                for j, (_samp, pred, actual) in enumerate(zip(_samps_in_batch, preds, actuals)):
                    figpath = os.path.join(figdir, f"{nm}-{_samp}.png")
                    ax = fig.add_subplot(1, 2, 1)
                    cb = ax.contourf(pred.cpu().detach().numpy())
                    ax.set_title("Predicted", fontsize=14)
                    fig.colorbar(cb)

                    ax = fig.add_subplot(1, 2, 2)
                    cb = ax.contourf(actual.cpu().detach().numpy())
                    ax.set_title("Actual", fontsize=14)
                    fig.colorbar(cb)

                    fig.savefig(figpath, bbox_inches="tight")
                    fig.clf()
        plt.close()
