from torch.utils.data import DataLoader
import lightning as L
import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from dateutil import parser
from itertools import product
from scipy.interpolate import interpn


def collate_fn(batch):
    new_batch = {}
    for k in batch[0].keys():
        new_batch[k] = [item[k] for item in batch]

    return new_batch


class ZEUSDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data["prompt"])


class ZEUSDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, test=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.inputs = ["Pointing", "Interf"]
        self.outputs = ["EspecH", "EspecL"]
        self.test = test

    def setup(self, stage: str):
        all_dat = defaultdict(list)

        df = pd.read_excel("/pscratch/sd/a/archis/processed_TA3_Dollar.xlsx", index_col=0)

        if self.test:
            cutoff = 850
        else:
            cutoff = 350

        all_rows = list(df.iterrows())[cutoff:]  # 350
        missing = []

        for idx, row in tqdm(all_rows, total=len(all_rows)):
            # find and move data
            date = row["Date"]
            run_num = row["Run No."]
            new_dict = {k.replace("(", "_").replace(")", "_").replace("=", "-"): v for k, v in dict(row).items()}
            # mlflow.log_params(new_dict)
            if parser.parse(row["Date"]) >= datetime(2023, 12, 8):
                parsed_date = parser.parse(row["Date"])
                source = os.path.join(self.data_dir, f"{parsed_date.month}{parsed_date.day:02d}_run{run_num}")
                skipped = False
                for subdir in self.inputs + self.outputs:  # os.listdir(source):
                    # if subdir in self.inputs + self.outputs:
                    filename = os.path.join(source, subdir, f"shot{row['Shot No.']}.tiff")
                    if not os.path.exists(filename):
                        skipped = True

                if not skipped:
                    for subdir in self.inputs + self.outputs:  # os.listdir(source):

                        filename = os.path.join(source, subdir, f"shot{row['Shot No.']}.tiff")
                        img = Image.open(filename)
                        all_dat[subdir].append(img)

                        if "espec" in subdir.casefold():
                            xx = 2160 - 256 * 2
                            dx = int(xx / 20)
                            dy = int(3840 / 50)
                            interp_points = list(
                                product(
                                    np.arange(256, 2160 - 256, xx / dx),
                                    np.arange(3840 / dy / 2, 3840 - 3840 / dy / 2 + 1, 3840 / dy),
                                )
                            )
                            interpd = interpn(
                                (np.arange(2160), np.arange(3840)), np.array(img), interp_points, method="linear"
                            ).reshape(dx, dy)
                            all_dat[subdir + "-downsampled"].append(torch.from_numpy(interpd).to(torch.float32))

                    prompts = []
                    for k, v in row.items():
                        if k in ["Date", "General comments ", "Run No.", "Shot No.", "notes"]:
                            pass
                        elif "nickname" in k:
                            pass
                        elif v != v:
                            prompts.append(f"The value of {k} was not available or missing.")
                        else:
                            prompts.append(f"The value of {k} was {v}.")

                    all_dat["prompt"].append(prompts)

        rng = np.random.default_rng(42)

        all_inds = np.arange(len(all_dat["EspecL-downsampled"]))
        self.train_inds = rng.choice(all_inds, int(0.8 * len(all_rows)), replace=False)
        self.val_inds = rng.choice(list(set(all_inds) - set(self.train_inds)), int(0.1 * len(all_rows)), replace=False)
        self.test_inds = list(set(all_inds) - set(self.train_inds) - set(self.val_inds))

        print(f"{len(self.train_inds)=}", f"{len(self.val_inds)=}", f"{len(self.test_inds)=}")

        self.train = ZEUSDataset(
            {
                k: [all_dat[k][ind] for ind in self.train_inds]
                for k in self.inputs + ["prompt", "EspecH-downsampled", "EspecL-downsampled"]
            }
        )
        self.val = ZEUSDataset(
            {
                k: [all_dat[k][ind] for ind in self.val_inds]
                for k in self.inputs + ["prompt", "EspecH-downsampled", "EspecL-downsampled"]
            }
        )
        self.test = ZEUSDataset(
            {
                k: [all_dat[k][ind] for ind in self.test_inds]
                for k in self.inputs + ["prompt", "EspecH-downsampled", "EspecL-downsampled"]
            }
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=collate_fn)  # , num_workers=64)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate_fn)  # , num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate_fn)  # , num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, collate_fn=collate_fn)
