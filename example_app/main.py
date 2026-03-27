#!/usr/bin/env python3


import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("--db_nodes", type=int, default=1)
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for the experiment")
args = parser.parse_args()

use_gpu = args.use_gpu

device = "GPU" if use_gpu else "CPU"
python_env = "/hpcwork/ro092286/smartsim/python/smartsim_cpu" if not use_gpu else "/hpcwork/ro092286/smartsim/python/smartsim_cuda-12"
queue = "c23g" if use_gpu else "c23ms"

print(f"Using device: {'GPU' if use_gpu else 'CPU'} (device={device}, python_env={python_env}, queue={queue})")

from smartsim.experiment import Experiment


exp_dir = "my_experiment"
if os.path.exists(exp_dir):
    print(f"Cleaning up previous experiment directory: {exp_dir}")
    shutil.rmtree(exp_dir)

exp = Experiment(name=exp_dir, launcher="slurm")

db = exp.create_database(
    port=6380,
    batch=True,
    db_nodes=args.db_nodes,
    interface="ib0",
    account="p0025821",
    time="04:00:00",
    queue=queue
)

db.set_run_arg("export", "ALL")


if exp.launcher == "slurm" or exp.launcher == "auto":
    db.batch_settings.add_preamble( # type: ignore
    [
#        "#SBATCH --exclusive",
        "#SBATCH --gres=gpu:1" if use_gpu else "",
        "#SBATCH --mem-per-cpu=100G",
        "source /hpcwork/ro092286/smartsim/set_env_claix23.sh",
        f"source {python_env}/bin/activate",
    ]
    )


exp.start(db, block=True, summary=True)


solver_rs = exp.create_run_settings(
    exe="./build/dummy_solver",
    #run_command="local",
    run_command="srun",
    run_args={
        "nodes": 1,
        "ntasks-per-node": 1,
        "cpus-per-task": 1
    }
)
solver_rs.update_env({"SSDB": "127.0.0.1:6380"})

solver_bs = exp.create_batch_settings(
    batch=True,
    time="04:00:00",
    account="p0025821",
    queue=queue
)

if exp.launcher == "slurm" or exp.launcher == "auto":
    solver_bs.add_preamble( # type: ignore
    [
        #"#SBATCH --exclusive",
        #"#SBATCH --gres=gpu:1" if use_gpu else "",
        "#SBATCH --mem-per-cpu=100G",
        #"source /hpcwork/ro092286/smartsim/set_env_claix23.sh",
        #f"source {python_env}/bin/activate",
        #f"export LD_LIBRARY_PATH={python_env}/lib:{python_env}/lib64:$LD_LIBRARY_PATH",
    ]
    )
    

solver = exp.create_model(name="solver", run_settings=solver_rs, batch_settings=solver_bs)

exp.start(solver)

# Print DB address for debugging
address = db.get_address()
print(f"DB address: {address}")  # e.g. ['node123.cluster:6380']

input("Press Enter to stop the experiment...")

exp.stop(solver)
exp.stop(db)