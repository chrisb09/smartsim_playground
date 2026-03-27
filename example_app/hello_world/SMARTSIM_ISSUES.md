# SmartSim Known Issues & Workarounds

## Issue 1: Misleading STATUS_FAILED after `exp.stop()`

### Problem
When calling `exp.stop(db_cluster)`, the JobManager logs show:
```
SmartSimStatus.STATUS_FAILED
```

This is **semantically incorrect** - the orchestrator was intentionally stopped, not failed. SmartSim should report `STATUS_CANCELLED` instead.

### Impact
- Confusing for users (looks like an error occurred)
- Makes it harder to distinguish actual failures from intentional shutdowns
- Can trigger false alerts in monitoring systems

### Proper Behavior
SmartSim should differentiate between:
- `STATUS_FAILED`: Process crashed or exited with error
- `STATUS_CANCELLED`: User called `exp.stop()` or used `scancel`

### Current Workaround
Ignore `STATUS_FAILED` messages that appear immediately after calling `exp.stop()`. They indicate successful termination.

---

## Issue 2: Redis Cluster Config Files Not Cleaned Up

### Problem
After stopping the orchestrator, Redis cluster configuration files remain:
```
exp_hello_world_db/orchestrator/nodes-orchestrator_0-6379.conf
exp_hello_world_db/orchestrator/nodes-orchestrator_1-6379.conf
exp_hello_world_db/orchestrator/nodes-orchestrator_2-6379.conf
```

Re-running the script causes:
```
[ERR] Node 134.61.46.136:6379 is not empty. Either the node already knows 
other nodes (check with CLUSTER NODES) or contains some key in database 0.
```

### Impact
- Cannot re-run experiments without manual cleanup
- Error-prone workflow (easy to forget cleanup)
- Debugging difficulty (stale state from previous runs)

### Proper Behavior
SmartSim should:
1. Clean up config files when `exp.stop()` is called
2. OR provide an `exp.cleanup()` method
3. OR check/clear stale configs in `exp.start()`

### Current Workaround
Manually delete the experiment directory before each run:
```bash
rm -rf exp_hello_world_db
python3 hello_world_db.py
```

Or programmatically:
```python
import shutil
if os.path.exists("exp_hello_world_db"):
    shutil.rmtree("exp_hello_world_db")
```

---

## Issue 3: Background Jobs Not Tracked

### Problem
When a batch orchestrator job fails during startup, it may remain in SLURM queue:
```bash
$ squeue --me
JOBID PARTITION     NAME     USER ST       TIME  NODES
64401196     c23ms orchestr ro092286  R       5:12      3
```

But SmartSim reports it as failed and moves on.

### Impact
- Wastes compute resources
- Confusing system state
- Requires manual `scancel`

### Proper Behavior
SmartSim should automatically cancel jobs when reporting STATUS_FAILED.

### Current Workaround
Always check for lingering jobs after failures:
```bash
squeue --me
scancel <JOBID>  # or scancel --me
```

---

## Recommendations for SmartSim Development Team

1. **Fix status reporting**: Map SIGTERM/intentional shutdown to `STATUS_CANCELLED`
2. **Implement cleanup hooks**: Remove stale config files in `stop()` or provide cleanup API
3. **Job lifecycle management**: Auto-cancel failed batch jobs
4. **Better error messages**: Distinguish between user actions and actual failures
