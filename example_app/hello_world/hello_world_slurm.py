# hello_world.py
from smartsim.experiment import Experiment

exp = Experiment("exp_hello_world_slurm", launcher="auto")
run = exp.create_run_settings(exe="echo", exe_args=["Hello World!"])
run.set_tasks(20)
run.set_tasks_per_node(20)

model = exp.create_model("hello_world", run)
exp.start(model, block=True, summary=True)

print(exp.get_status(model))