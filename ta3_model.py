from parsl.app.app import python_app


def training_run(rid, cfg, log_dir):
    import os
    import lightning as L
    from model import ZEUSLightningModule, CustomWriter
    from data import ZEUSDataModule
    from lightning.pytorch.loggers import MLFlowLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    ZDM = ZEUSDataModule(
        os.path.join("/pscratch/sd/a/archis", "TA3_Dollar_2023_backup"), data_params=cfg["data"], test=False
    )

    cfg["model"] = cfg["model"] | {"nx": ZDM.nx, "ny": ZDM.ny}

    ZLM = ZEUSLightningModule(
        learning_rate=cfg["fitter"]["learning_rate"],
        batch_size=cfg["data"]["batch_size"],
        model_params=cfg["model"],
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
        # devices=[0, 1, 2, 3],
        devices=1,
        accelerator="gpu",
        logger=mlf_logger,
        max_epochs=500,
        callbacks=[early_stopping, checkpoint, prediction_writer],
        default_root_dir=os.path.join(os.environ["SCRATCH"], "lightning"),
    )

    trainer.fit(model=ZLM, datamodule=ZDM)
    trainer.predict(model=ZLM, datamodule=ZDM, return_predictions=False)

    return f"{rid=} completed successfully"


if __name__ == "__main__":
    from itertools import product
    from misc import setup_parsl, export_run
    import yaml, mlflow, os, shutil

    parsl = True
    if parsl:
        setup_parsl("local", 4, max_blocks=3)
        training_run = python_app(training_run)

    bss = [4, 8, 16, 32]
    lrs = [1e-3, 1e-4, 1e-5]
    num_channels = [4, 8, 16, 32]

    alls = product(lrs, bss, num_channels)

    res, rids = [], []
    for lr, bs, ncs in alls:
        with open("config.yaml", "r") as fi:
            cfg = yaml.safe_load(fi)
        cfg["fitter"]["learning_rate"] = lr
        cfg["fitter"]["batch_size"] = bs
        cfg["model"]["num_channels"] = ncs
        cfg["mlflow"]["run"] = f"{ncs=}-{lr=}-{bs=}"

        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as run:
            mlflow.log_params(cfg)

        log_dir = os.path.join(os.environ["SCRATCH"], "lightning", f"{run.info.run_id}")
        os.makedirs(log_dir, exist_ok=True)

        res.append(training_run(run.info.run_id, cfg, log_dir))
        rids.append(run.info.run_id)

    for run_id, rr in zip(rids, res):
        if parsl:
            print(rr.result())
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_artifacts(log_dir := os.path.join(os.environ["SCRATCH"], "lightning", f"{run_id}"))

        export_run(run_id)
        shutil.rmtree(log_dir)
