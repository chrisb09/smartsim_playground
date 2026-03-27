# run_db_batch.py
from smartsim.experiment import Experiment
import os
import shutil

#os.environ["LD_LIBRARY_PATH"] = "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/OpenSSL/1.1/lib:$LD_LIBRARY_PATH"

#os.system("module load OpenSSL/1.1")

# Clean up previous experiment directory to avoid "Node not empty" errors
exp_dir = "exp_hello_world_db"
if os.path.exists(exp_dir):
    print(f"Cleaning up previous experiment directory: {exp_dir}")
    shutil.rmtree(exp_dir)

exp = Experiment(exp_dir, launcher="auto")
db_cluster = exp.create_database(db_nodes=3, # 2 is not allowed
                                 db_port=6780,
                                 batch=True,
                                 time="00:10:00",
                                 interface="ib0",
                                 account="p0025821",
                                 queue="c23ms")

db_cluster.batch_settings.add_preamble(
    [
        "#SBATCH --exclusive",
        "python -c \"import ssl; print(ssl.OPENSSL_VERSION)\"",
        "module load Python/3.11.5",
        "module load OpenSSL/1.1",
        "export LD_LIBRARY_PATH=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Python/3.11.5-GCCcore-13.2.0/lib:/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/OpenSSL/1.1/lib:$LD_LIBRARY_PATH",
        "python -c \"import ssl; print(ssl.OPENSSL_VERSION)\""
    ]
)

print(db_cluster.batch_settings.batch_cmd)
print(db_cluster.batch_settings.batch_args)

db_cluster.set_run_arg("export", "ALL")


exp.start(db_cluster)

print(f"Orchestrator launched on nodes: {db_cluster.hosts}")
print(f"Database addresses: {db_cluster.get_address()}")

# Wait for user input before stopping (for testing)
input("Press Enter to stop the orchestrator...")

exp.stop(db_cluster)