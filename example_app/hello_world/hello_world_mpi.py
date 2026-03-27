from smartsim.experiment import Experiment

exp = Experiment("exp_hello_world_mpi", launcher="local")
mpi_settings = exp.create_run_settings(exe="echo",
                                       exe_args=["Hello World!"],
                                       run_command="mpirun")
mpi_settings.set_tasks(4)

mpi_model = exp.create_model("hello_world", mpi_settings)

exp.start(mpi_model, block=True)
print(exp.get_status(mpi_model))