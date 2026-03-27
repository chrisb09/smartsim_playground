# hello_ensemble.py
from smartsim.experiment import Experiment

exp = Experiment("exp_hello_world_batch", launcher="auto", exp_path="/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch")

# define resources for all ensemble members
batch = exp.create_batch_settings(nodes=2, time="00:10:00", account="p0025821",
                                  batch_args={"ntasks": "4",
                                              "cpus-per-task": "1"})
batch.set_queue("c23ms")

# define how each member should run
run = exp.create_run_settings(exe="./simple.sh", exe_args=[])
run.set_tasks_per_node(2)
run.set_cpus_per_task(1)

ensemble = exp.create_ensemble("hello_world_batch_ensemble",
                               batch_settings=batch,
                               run_settings=run,
                               replicas=1)
exp.start(ensemble, block=True, summary=True)

print(exp.get_status(ensemble))