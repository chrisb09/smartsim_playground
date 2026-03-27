#!/bin/bash

#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/d8c29b0/model/solver/solver.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/d8c29b0/model/solver/solver.err
#SBATCH --job-name=solver-DGMPPV55HU31
#SBATCH --nodes=1
#SBATCH --partition=c23ms
#SBATCH --time=04:00:00
#SBATCH --account=p0025821
#SBATCH --mem-per-cpu=100G

cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/solver ; /usr/local_host/bin/srun --output /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/d8c29b0/model/solver/solver/solver.out --error /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/d8c29b0/model/solver/solver/solver.err --job-name solver-DGMPPV55L4K9 --export ALL,SSDB=134.61.46.154:6380,SR_DB_TYPE=Standalone --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/build/dummy_solver &

wait
