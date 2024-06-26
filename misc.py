import mlflow, os, boto3, tempfile, flatdict
from mlflow_export_import.run.export_run import RunExporter


def log_params(cfg):
    flattened_dict = dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    if num_entries > 100:
        num_batches = num_entries % 100
        fl_list = list(flattened_dict.items())
        for i in range(num_batches):
            end_ind = min((i + 1) * 100, num_entries)
            trunc_dict = {k: v for k, v in fl_list[i * 100 : end_ind]}
            mlflow.log_params(trunc_dict)
    else:
        mlflow.log_params(flattened_dict)


def upload_dir_to_s3(local_directory: str, bucket: str, destination: str, run_id: str, prefix="individual", step=0):
    """
    Uploads directory to s3 bucket for ingestion into mlflow on remote / cloud side

    This requires you to have permission to access the s3 bucket

    :param local_directory:
    :param bucket:
    :param destination:
    :param run_id:
    :return:
    """
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            client.upload_file(local_path, bucket, s3_path)

    with open(os.path.join(local_directory, f"ingest-{run_id}.txt"), "w") as fi:
        fi.write("ready")

    if prefix == "individual":
        fname = f"ingest-{run_id}.txt"
    else:
        fname = f"{prefix}-{run_id}-{step}.txt"

    client.upload_file(os.path.join(local_directory, f"ingest-{run_id}.txt"), bucket, fname)


def export_run(run_id, prefix="individual", step=0):
    run_exp = RunExporter(mlflow_client=mlflow.MlflowClient())
    with tempfile.TemporaryDirectory() as td2:
        run_exp.export_run(run_id, td2)
        upload_dir_to_s3(td2, "remote-mlflow-staging", f"artifacts/{run_id}", run_id, prefix, step)


def setup_parsl(parsl_provider="local", num_gpus=4, max_blocks=3):
    import parsl
    from parsl.config import Config
    from parsl.providers import SlurmProvider, LocalProvider
    from parsl.launchers import SrunLauncher
    from parsl.executors import HighThroughputExecutor

    if parsl_provider == "local":

        print(f"Using local provider, ignoring {max_blocks=}")

        this_provider = LocalProvider
        provider_args = dict(
            worker_init="source /pscratch/sd/a/archis/venvs/zeus-gpu/bin/activate; \
                    module load cudnn/8.9.3_cuda12.lua; \
                    export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/zeus/'; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow'; \
                    export MLFLOW_EXPORT=True",
            init_blocks=1,
            max_blocks=1,
        )

        htex = HighThroughputExecutor(
            available_accelerators=["0,1,2,3"],
            label="zeus",
            provider=this_provider(**provider_args),
            cpu_affinity="block",
        )
        print(f"{htex.workers_per_node=}")

    elif "gpu" in parsl_provider:

        this_provider = SlurmProvider
        sched_args = ["#SBATCH -C gpu", "#SBATCH --ntasks-per-node 4"]
        if "debug" in parsl_provider:
            sched_args += ["#SBATCH --qos=debug"]
            walltime = "00:10:00"
        else:
            # sched_args = ["#SBATCH -C gpu", "#SBATCH --qos=regular", "#SBATCH --ntasks-per-node 4"]
            sched_args += ["#SBATCH --qos=regular"]
            walltime = "12:00:00"
        provider_args = dict(
            partition=None,
            account="m4490_g",
            scheduler_options="\n".join(sched_args),
            worker_init="export SLURM_CPU_BIND='cores';\
                    source /pscratch/sd/a/archis/venvs/zeus-gpu/bin/activate; \
                    export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/zeus/'; \
                    module load cudnn/8.9.3_cuda12.lua; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';\
                    export MLFLOW_EXPORT=True",
            launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 32"),
            walltime=walltime,
            cmd_timeout=120,
            nodes_per_block=1,
            # init_blocks=1,
            max_blocks=max_blocks,
        )

        htex = HighThroughputExecutor(
            available_accelerators=["0,1,2,3"],
            label="zeus",
            provider=this_provider(**provider_args),
            cpu_affinity="block",
            # max_workers_per_node=1
        )
        print(f"{htex.workers_per_node=}")

    config = Config(executors=[htex], retries=4)

    # load the Parsl config
    parsl.load(config)
