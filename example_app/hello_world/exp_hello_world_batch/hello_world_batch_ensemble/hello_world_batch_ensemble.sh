#!/bin/bash

#SBATCH --output=/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch/.smartsim/telemetry/exp_hello_world_batch/2db7451/ensemble/hello_world_batch_ensemble/hello_world_batch_ensemble.out
#SBATCH --error=/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch/.smartsim/telemetry/exp_hello_world_batch/2db7451/ensemble/hello_world_batch_ensemble/hello_world_batch_ensemble.err
#SBATCH --job-name=hello_world_batch_ensemble-DG2RYBXARPJH
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --account=p0025821
#SBATCH --partition=c23ms

cd /hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch/hello_world_batch_ensemble/hello_world_batch_ensemble_0 ; /usr/local_host/bin/srun --output /hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch/.smartsim/telemetry/exp_hello_world_batch/2db7451/ensemble/hello_world_batch_ensemble/hello_world_batch_ensemble_0/hello_world_batch_ensemble_0.out --error /hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_batch/.smartsim/telemetry/exp_hello_world_batch/2db7451/ensemble/hello_world_batch_ensemble/hello_world_batch_ensemble_0/hello_world_batch_ensemble_0.err --job-name hello_world_batch_ensemble_0-DG2RYBXAUMNI --export ALL,SSKEYOUT=hello_world_batch_ensemble_0 --ntasks-per-node=2 --cpus-per-task=1 /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/simple.sh &

wait
