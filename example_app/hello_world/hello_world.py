from smartsim.experiment import Experiment

exp = Experiment("exp_hello_world", launcher="local")

settings = exp.create_run_settings("echo", exe_args=["Hello World"])
model = exp.create_model("hello_world", settings)

exp.start(model, block=True)
print(exp.get_status(model))