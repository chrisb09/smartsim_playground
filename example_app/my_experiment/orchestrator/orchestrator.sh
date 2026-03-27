#!/bin/bash

#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/6d8c5f0/database/orchestrator/orchestrator.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/6d8c5f0/database/orchestrator/orchestrator.err
#SBATCH --job-name=orchestrator-DGMPPDZBFY42
#SBATCH --nodes=1
#SBATCH --partition=c23ms
#SBATCH --time=04:00:00
#SBATCH --account=p0025821

#SBATCH --mem-per-cpu=100G
source /hpcwork/ro092286/smartsim/set_env_claix23.sh
source /hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/activate

cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/orchestrator ; /usr/local_host/bin/srun --output /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/6d8c5f0/database/orchestrator/orchestrator_0/orchestrator_0.out --error /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/my_experiment/.smartsim/telemetry/my_experiment/6d8c5f0/database/orchestrator/orchestrator_0/orchestrator_0.err --job-name orchestrator_0-DGMPPDZBHNUG --ntasks=1 --ntasks-per-node=1 --export=ALL /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/python3 -m smartsim._core.entrypoints.redis +orc-exe=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/bin/redis-server +conf-file=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/config/redis.conf +rai-module --loadmodule /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so +name=orchestrator_0 +port=6380 +ifname=ib0 &

wait
