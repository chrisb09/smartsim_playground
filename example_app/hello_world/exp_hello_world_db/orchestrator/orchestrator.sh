#!/bin/bash

#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_db/.smartsim/telemetry/exp_hello_world_db/c69f3c4/database/orchestrator/orchestrator.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_db/.smartsim/telemetry/exp_hello_world_db/c69f3c4/database/orchestrator/orchestrator.err
#SBATCH --job-name=orchestrator-DG4L99BDZ9OA
#SBATCH --nodes=3
#SBATCH --partition=c23ms
#SBATCH --time=00:10:00
#SBATCH --account=p0025821
#SBATCH --exclusive
python -c "import ssl; print(ssl.OPENSSL_VERSION)"
module load Python/3.11.5
module load OpenSSL/1.1
export LD_LIBRARY_PATH=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Python/3.11.5-GCCcore-13.2.0/lib:/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/OpenSSL/1.1/lib:$LD_LIBRARY_PATH
python -c "import ssl; print(ssl.OPENSSL_VERSION)"

cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_db/orchestrator ; /usr/local_host/bin/srun --output /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_db/.smartsim/telemetry/exp_hello_world_db/c69f3c4/database/orchestrator/orchestrator_0/orchestrator_0.out --error /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/example_app/hello_world/exp_hello_world_db/.smartsim/telemetry/exp_hello_world_db/c69f3c4/database/orchestrator/orchestrator_0/orchestrator_0.err --job-name orchestrator_0-DG4L99BE4GFL --ntasks=1 --ntasks-per-node=1 --export=ALL /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/bin/python3 -m smartsim._core.entrypoints.redis +orc-exe=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/bin/redis-server +conf-file=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/config/redis.conf +rai-module --loadmodule /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so +name=orchestrator_0 +port=6379 +ifname=ib0 +cluster : --ntasks=1 --ntasks-per-node=1 --export=ALL --job-name orchestrator_0-DG4L99BE4GFL /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/bin/python3 -m smartsim._core.entrypoints.redis +orc-exe=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/bin/redis-server +conf-file=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/config/redis.conf +rai-module --loadmodule /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so +name=orchestrator_1 +port=6379 +ifname=ib0 +cluster : --ntasks=1 --ntasks-per-node=1 --export=ALL --job-name orchestrator_0-DG4L99BE4GFL /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/bin/python3 -m smartsim._core.entrypoints.redis +orc-exe=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/bin/redis-server +conf-file=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/config/redis.conf +rai-module --loadmodule /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/python/smartsim/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so +name=orchestrator_2 +port=6379 +ifname=ib0 +cluster &

wait
