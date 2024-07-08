from itertools import product
import yaml, mlflow, os, time
import numpy as np
from misc import log_params
from train_ta3_model import training_run


if __name__ == "__main__":

    bss = [4, 8, 16, 32]
    lrs = [1e-3, 1e-4, 1e-5]
    num_channels = [4, 8, 16, 32]
    num_layers = [3, 4, 5, 6]

    alls = list(product(lrs, bss, num_channels, num_layers))
    np.random.shuffle(alls)
    for lr, bs, ncs, nls in alls[:1]:
        with open("config.yaml", "r") as fi:
            cfg = yaml.safe_load(fi)
        cfg["fitter"]["learning_rate"] = lr
        cfg["fitter"]["batch_size"] = bs
        cfg["model"]["num_channels"] = ncs
        cfg["model"]["num_layers"] = nls
        cfg["mlflow"]["run"] = f"{ncs=}-{lr=}-{bs=}-{nls=}"

        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as run:
            log_params(cfg)
            mlflow.set_tag("status", "QUEUED")

        log_dir = os.path.join(os.environ["SCRATCH"], "lightning", f"{run.info.run_id}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.yaml"), "w") as fi:
            yaml.dump(cfg, fi)

        if cfg["debug"]:
            training_run(run.info.run_id)
        else:
            with open("nersc-gpu.sh", "r") as fh:
                base_job = fh.read()

            with open(os.path.join(os.getcwd(), "queue_ta3_training_run.sh"), "w") as job_file:
                job_file.write(base_job)
                job_file.writelines("\n")
                job_file.writelines(f"srun python train_ta3_model.py --run_id {run.info.run_id}")

            os.system(f"sbatch queue_ta3_training_run.sh")
            time.sleep(1)

    os.system("sqs")
    print("runs queued")
