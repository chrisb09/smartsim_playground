import os, sys, time
from smartsim.experiment import Experiment
exp = Experiment("db_3053170_smartsim_b2000000", launcher="local")
db = exp.create_database(port=7246, interface="lo", db_nodes=1, threads_per_queue=4, intra_op_threads=1, inter_op_threads=1)
exp.generate(db, overwrite=True)
exp.start(db)
print(f"DB_READY:127.0.0.1:{7246}")
sys.stdout.flush()
time.sleep(600)
