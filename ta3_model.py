from parsl.app.app import python_app


@python_app
def training_run(hparams):
    import os, yaml
    import lightning as L
    from misc import export_run
    from model import ZEUSLightningModule
    from data import ZEUSDataModule
    from lightning.pytorch.loggers import MLFlowLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.utilities import rank_zero_only

    ZLM = ZEUSLightningModule(learning_rate=hparams["learning_rate"], batch_size=hparams["batch_size"])
    ZDM = ZEUSDataModule(
        os.path.join("/pscratch/sd/a/archis", "TA3_Dollar_2023_backup"),
        batch_size=hparams["batch_size"],
        test=False,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=hparams["mlflow"]["experiment"],
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        log_model=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    trainer = L.Trainer(
        # devices=[0, 1, 2, 3],
        devices=1,
        accelerator="gpu",
        logger=mlf_logger,
        max_epochs=40,
        callbacks=[early_stopping, checkpoint],
        default_root_dir=os.path.join(os.environ["SCRATCH"], "lightning"),
    )

    # write config
    if trainer.logger.run_id is not None:
        config_dir = os.path.join(os.environ["SCRATCH"], "lightning", "configs")
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, "w") as fi:
            yaml.dump(hparams, fi)

    trainer.fit(model=ZLM, datamodule=ZDM)

    trainer.logger.log_hyperparams(hparams)

    # append run_id to completed_run_ids.txt
    if trainer.logger.run_id is not None:
        trainer.logger.experiment.log_artifacts(trainer.logger.run_id, config_dir)
        export_run(trainer.logger.run_id)


if __name__ == "__main__":
    from itertools import product
    from misc import setup_parsl

    setup_parsl("gpu", 4, max_blocks=3)

    bss = [4, 8, 16, 32][::-1]
    lrs = [1e-3, 1e-4, 1e-5]

    alls = product(lrs, bss)

    res = []
    for lr, bs in alls:
        hparams = {"learning_rate": lr, "batch_size": bs, "mlflow": {"experiment": "zeus-model"}}
        res.append(training_run(hparams))

    for r in res:
        print(r.result())
