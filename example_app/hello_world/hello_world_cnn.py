# simple_torch_inference.py
import time
start_time = time.time()
import io
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for the experiment")
args = parser.parse_args()

print(f"Basic imports done in {time.time() - start_time:.2f} seconds")
import_torch_start = time.time()
import torch
import torch.nn as nn
print(f"Torch imported in {time.time() - import_torch_start:.2f} seconds")
import_redis_start = time.time()
from smartredis import Client
from smartsim.experiment import Experiment
print(f"Smartredis import done in {time.time() - import_redis_start:.2f} seconds")


use_gpu = args.use_gpu

device = "GPU" if use_gpu else "CPU"
python_env = "/hpcwork/ro092286/smartsim/python/smartsim_cpu" if not use_gpu else "/hpcwork/ro092286/smartsim/python/smartsim_cuda-12"
queue = "c23g" if use_gpu else "c23ms"

print(f"Using device: {'GPU' if use_gpu else 'CPU'} (device={device}, python_env={python_env}, queue={queue})")


exp_dir = "exp_hello_world_cnn"
if os.path.exists(exp_dir):
    print(f"Cleaning up previous experiment directory: {exp_dir}")
    shutil.rmtree(exp_dir)


start_create_exp = time.time()
exp = Experiment(exp_dir, launcher="slurm")
print(f"Experiment creation done in {time.time() - start_create_exp:.2f} seconds")
start_db_create = time.time()
db = exp.create_database(port=6780,
                         interface="ib0",
                         batch=True,
                         time="00:10:00",
                         account="p0025821",
                         queue=queue
                         )


db.set_run_arg("export", "ALL")

print(f"Database creation done in {time.time() - start_db_create:.2f} seconds")

if exp.launcher == "slurm" or exp.launcher == "auto":
    db.batch_settings.add_preamble( # type: ignore
    [
        "#SBATCH --exclusive",
        "#SBATCH --gres=gpu:1" if use_gpu else "",
        "source /hpcwork/ro092286/smartsim/set_env_claix23.sh",
        f"source {python_env}/bin/activate",
        f"export LD_LIBRARY_PATH={python_env}/lib:{python_env}/lib64:$LD_LIBRARY_PATH",
    ]
    )


print("DDDD")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

torch_model = Net()
example_forward_input = torch.rand(1, 1, 3, 3)
module = torch.jit.trace(torch_model, example_forward_input)
model_buffer = io.BytesIO()
torch.jit.save(module, model_buffer)


print("EEEE")

exp.start(db, block=True, summary=True)

print("FFFF")

print(exp.db_identifiers)


print("Experiment started (including database)")

address = db.get_address()[0]
client = Client(address=address, cluster=False)

client.put_tensor("input", example_forward_input.numpy())
client.set_model("cnn", model_buffer.getvalue(), "TORCH", device=device)
client.run_model("cnn", inputs=["input"], outputs=["output"])
output = client.get_tensor("output")
print(f"Prediction: {output}")

exp.stop(db)