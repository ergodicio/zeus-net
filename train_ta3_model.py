import os, argparse, yaml
import lightning as L
from model import ZEUSLightningModule, CustomWriter
from data import ZEUSDataModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def training_run(rid):

    # download config
    log_dir = os.path.join(os.environ["SCRATCH"], "lightning", f"{rid}")

    with open(os.path.join(log_dir, "config.yaml"), "r") as fi:
        cfg = yaml.safe_load(fi)

    ZDM = ZEUSDataModule(
        os.path.join("/pscratch/sd/a/archis", "TA3_Dollar_2023_backup"), data_params=cfg["data"], test=cfg["debug"]
    )

    cfg["model"] = cfg["model"] | {"nx": ZDM.nx, "ny": ZDM.ny}

    ZLM = ZEUSLightningModule(
        learning_rate=cfg["fitter"]["learning_rate"],
        model_params=cfg["model"],
        run_id=rid,
        log_dir=log_dir,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=cfg["mlflow"]["experiment"],
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        log_model=True,
        run_id=rid,
    )
    early_stopping = EarlyStopping(monitor="val-loss", patience=32)
    checkpoint = ModelCheckpoint(monitor="val-loss", save_top_k=1, mode="min")
    prediction_writer = CustomWriter(write_interval="epoch", output_dir=log_dir)
    trainer = L.Trainer(
        devices=[0, 1, 2, 3],
        # devices=1,
        accelerator="gpu",
        logger=mlf_logger,
        max_epochs=2 if cfg["debug"] else 1000,
        callbacks=[early_stopping, checkpoint, prediction_writer],
        default_root_dir=os.path.join(os.environ["SCRATCH"], "lightning"),
    )

    trainer.fit(model=ZLM, datamodule=ZDM)
    trainer.predict(model=ZLM, datamodule=ZDM, return_predictions=False)

    return f"{rid=} completed successfully"


if __name__ == "__main__":
    # arg parse first arg as run_id

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--run_id", type=str, help="Run ID")
    args = parser.parse_args()
    run_id = args.run_id

    training_run(run_id)
