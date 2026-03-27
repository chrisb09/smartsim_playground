#!/bin/bash

#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_cnn/.smartsim/telemetry/exp_hello_world_cnn/ea4f553/database/orchestrator/orchestrator.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_cnn/.smartsim/telemetry/exp_hello_world_cnn/ea4f553/database/orchestrator/orchestrator.err
#SBATCH --job-name=orchestrator-DGL2K9OQ4XVU
#SBATCH --nodes=1
#SBATCH --partition=c23ms
#SBATCH --time=00:10:00
#SBATCH --account=p0025821
#SBATCH --exclusive

source /hpcwork/ro092286/smartsim/set_env_claix23.sh
source /hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/activate
export LD_LIBRARY_PATH=/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib:/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib64:$LD_LIBRARY_PATH

cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_cnn/orchestrator ; /usr/local_host/bin/srun --output /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_cnn/.smartsim/telemetry/exp_hello_world_cnn/ea4f553/database/orchestrator/orchestrator_0/orchestrator_0.out --error /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_cnn/.smartsim/telemetry/exp_hello_world_cnn/ea4f553/database/orchestrator/orchestrator_0/orchestrator_0.err --job-name orchestrator_0-DGL2K9OQ6O9Z --ntasks=1 --ntasks-per-node=1 --export=ALL /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/python3 -m smartsim._core.entrypoints.redis +orc-exe=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/bin/redis-server +conf-file=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/config/redis.conf +rai-module --loadmodule /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so +name=orchestrator_0 +port=6780 +ifname=ib0 &

wait
