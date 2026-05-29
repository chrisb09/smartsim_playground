import os
import subprocess
import time
import threading
import signal
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_TEST_DIR = BASE_DIR / "module_test"
TRAIN_MODELS_DIR = BASE_DIR / "mini_app" / "train_models" / "model_a"

# Configurations
PROVIDERS = ["AIX", "PHYDLL", "SMARTSIM"]
DEVICES = ["CPU", "GPU"]
MODELS = ["perfect", "transformer"]
WORKLOADS = [
    (1, 1), # (steps, clients)
    (5, 1),
    (5, 2)
]

DEFAULT_GPU_ID = 3
GPU_RANKS_TO_EXCLUDE = int(os.environ.get("GPU_RANKS_TO_EXCLUDE", "1"))

RESULTS = []

class ResourceMonitor:
    def __init__(self, target_gpu=None):
        self.target_gpu = target_gpu
        self.max_cpu_solver_kb = 0
        self.max_cpu_ml_kb = 0
        self.max_cpu_other_kb = 0
        self.max_cpu_total_kb = 0
        self.max_gpu_mem_mb = 0
        self.running = False
        self.root_pid = None

    def _read_proc_env(self, pid):
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                entries = f.read().split(b"\0")
            env = {}
            for entry in entries:
                if not entry or b"=" not in entry:
                    continue
                key, val = entry.split(b"=", 1)
                env[key.decode(errors="ignore")] = val.decode(errors="ignore")
            return env
        except Exception:
            return {}

    def _read_proc_cmdline(self, pid):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().split(b"\0")
            parts = [p.decode(errors="ignore") for p in raw if p]
            return " ".join(parts)
        except Exception:
            return ""

    def _is_excluded_gpu_rank(self, env):
        appnum = env.get("OMPI_COMM_WORLD_APPNUM")
        rank = env.get("OMPI_COMM_WORLD_RANK")
        if appnum is None or rank is None:
            return False
        try:
            return int(appnum) == 1 and int(rank) < GPU_RANKS_TO_EXCLUDE
        except ValueError:
            return False

    def _classify_process(self, cmdline):
        lowered = cmdline.lower()
        if "module_test_solver" in lowered:
            return "solver"
        if (
            "redis-server" in lowered
            or "redisai" in lowered
            or "driver.py" in lowered
            or "phydll_dl_client" in lowered
            or "dl_client" in lowered
        ):
            return "ml"
        return "other"

    def _get_descendant_pids(self):
        if not self.root_pid:
            return []
        try:
            ppid_map = {}
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = entry
                try:
                    with open(f"/proc/{pid}/stat", "r") as f:
                        stat = f.read().split()
                    if len(stat) >= 4:
                        ppid = stat[3]
                        ppid_map.setdefault(ppid, []).append(pid)
                except Exception:
                    continue

            root = str(self.root_pid)
            stack = [root]
            seen = set()
            pids = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                pids.append(current)
                children = ppid_map.get(current, [])
                stack.extend(children)

            filtered = []
            for pid in pids:
                env = self._read_proc_env(pid)
                if self._is_excluded_gpu_rank(env):
                    continue
                filtered.append(pid)
            return filtered
        except Exception:
            return []

    def get_group_memory(self):
        if not self.root_pid:
            return 0, 0, 0, 0
        try:
            cmd = ["ps", "-e", "-o", "pid,rss", "--no-headers"]
            out = subprocess.check_output(cmd, text=True)
            pid_set = set(self._get_descendant_pids())
            if not pid_set:
                return 0, 0, 0, 0
            solver_rss = 0
            ml_rss = 0
            other_rss = 0
            for line in out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] in pid_set:
                    pid = parts[0]
                    rss_kb = int(parts[1])
                    cmdline = self._read_proc_cmdline(pid)
                    bucket = self._classify_process(cmdline)
                    if bucket == "solver":
                        solver_rss += rss_kb
                    elif bucket == "ml":
                        ml_rss += rss_kb
                    else:
                        other_rss += rss_kb
            total_rss = solver_rss + ml_rss + other_rss
            return solver_rss, ml_rss, other_rss, total_rss
        except Exception:
            return 0, 0, 0, 0

    def get_gpu_memory(self):
        if self.target_gpu is None:
            return 0
        try:
            pids = set(self._get_descendant_pids())
            cmd = [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.target_gpu),
            ]
            out = subprocess.check_output(cmd, text=True)
            total_mb = 0
            matched = False
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and (not pids or parts[0] in pids):
                    total_mb += int(parts[1])
                    matched = matched or (parts[0] in pids)
            if not pids or not matched:
                if os.environ.get("GPU_MEM_FALLBACK_TOTAL", "0") == "1":
                    cmd = [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self.target_gpu),
                    ]
                    out = subprocess.check_output(cmd, text=True)
                    return int(out.strip().splitlines()[0])
                return 0
            return total_mb
        except Exception:
            return 0

    def monitor(self):
        while self.running:
            solver_rss, ml_rss, other_rss, total_rss = self.get_group_memory()
            gpu_mem = self.get_gpu_memory()
            
            if solver_rss > self.max_cpu_solver_kb:
                self.max_cpu_solver_kb = solver_rss
            if ml_rss > self.max_cpu_ml_kb:
                self.max_cpu_ml_kb = ml_rss
            if other_rss > self.max_cpu_other_kb:
                self.max_cpu_other_kb = other_rss
            if total_rss > self.max_cpu_total_kb:
                self.max_cpu_total_kb = total_rss
            if gpu_mem > self.max_gpu_mem_mb:
                self.max_gpu_mem_mb = gpu_mem
            
            time.sleep(0.5) # Polling interval

    def start(self, root_pid):
        self.root_pid = root_pid
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()

def run_command(cmd, env, target_gpu=None):
    monitor = ResourceMonitor(target_gpu)
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd, env=env, cwd=str(MODULE_TEST_DIR), 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            text=True, start_new_session=True
        )
        
        monitor.start(process.pid)
        
        try:
            stdout, stderr = process.communicate(timeout=1200)
            duration = time.time() - start_time
            success = process.returncode == 0
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGTERM)
            stdout, stderr = process.communicate()
            return False, 1200, 0, 0, 0, 0, 0, "TIMEOUT", "Execution timed out"
        finally:
            monitor.stop()

        # Improved summary extraction: find numeric output ranks
        summary_lines = []
        rank_pattern = re.compile(r'^Rank\s+\d+:.*')
        
        found_header = False
        for line in output.splitlines():
            line = line.strip()
            if "Gathered outputs from all ranks:" in line:
                found_header = True
                continue
            if found_header:
                if rank_pattern.match(line):
                    summary_lines.append(line)
                elif summary_lines and not line: # Stop on first empty line after ranks
                    break
                elif summary_lines and ("Finalize" in line or "Solver done" in line): # Stop on footer
                    break
        
        result_summary = " | ".join(summary_lines) if summary_lines else "N/A"
        
        return (
            success,
            duration,
            monitor.max_cpu_solver_kb / 1024.0,
            monitor.max_cpu_ml_kb / 1024.0,
            monitor.max_cpu_other_kb / 1024.0,
            monitor.max_cpu_total_kb / 1024.0,
            monitor.max_gpu_mem_mb,
            result_summary,
            output,
        )
    except Exception as e:
        return False, 0, 0, 0, 0, 0, 0, "ERROR", str(e)

def update_toml(provider, device, model_name):
    toml_path = MODULE_TEST_DIR / f"config_{provider.lower()}_{device.lower()}.toml"
    suffix = "cuda" if device == "GPU" else "cpu"
    model_file = TRAIN_MODELS_DIR / f"{model_name}_{suffix}.pt"
    
    with open(toml_path, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if line.strip().startswith("model_file =") or line.strip().startswith("model_path ="):
            key = "model_file" if line.strip().startswith("model_file") else "model_path"
            new_lines.append(f'{key} = "{model_file}"\n')
        else:
            new_lines.append(line)
    
    # Ensure SmartSim timeouts
    if provider == "SMARTSIM" and not any("command_timeout" in l for l in new_lines):
        for i, line in enumerate(new_lines):
            if "[provider]" in line:
                new_lines.insert(i+1, "command_timeout = 900\n")
                new_lines.insert(i+2, "socket_timeout = 900\n")
                new_lines.insert(i+3, "model_timeout = 900000\n")
                break

    with open(toml_path, "w") as f:
        f.writelines(new_lines)

# Header
print(f"| {'Provider':<9} | {'Dev':<4} | {'Model':<7} | {'St/Cl':<5} | {'Stat':<2} | {'Time':<6} | {'CPU_S':<7} | {'CPU_M':<7} | {'CPU_O':<7} | {'CPU_T':<7} | {'GPU(MB)':<7} | {'Results'}")
print(f"|{'-'*11}|{'-'*6}|{'-'*9}|{'-'*7}|{'-'*6}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*40}")

ss_port = 7200

for provider in PROVIDERS:
    for device in DEVICES:
        for model in MODELS:
            update_toml(provider, device, model)
            for steps, clients in WORKLOADS:
                env = os.environ.copy()
                env["PROVIDER"] = provider
                env["DEVICE"] = device
                env["STEPS"] = str(steps)
                env["CLIENTS"] = str(clients)
                env["COMPILE"] = "0"
                env["MODEL"] = model
                
                target_gpu = None
                if provider == "SMARTSIM":
                    env["MLCOUPLING_SMARTSIM_NODES"] = "1"
                    env["MLCOUPLING_SMARTSIM_NUM_GPUS"] = "1"
                    env["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_GPU_ID)
                    env["SS_PORT"] = str(ss_port)
                    ss_port += 1
                    if device == "GPU":
                        target_gpu = DEFAULT_GPU_ID
                elif device == "GPU":
                    env["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_GPU_ID)
                    target_gpu = DEFAULT_GPU_ID

                success, duration, cpu_solver_mb, cpu_ml_mb, cpu_other_mb, cpu_total_mb, gpu_mb, summary, full_log = run_command(["./run.sh"], env, target_gpu)
                
                status = "✅" if success else "❌"
                print(
                    f"| {provider:<9} | {device:<4} | {model:<7} | {steps}/{clients:<3} | {status:<2} | {duration:>5.1f}s | "
                    f"{cpu_solver_mb:>7.1f} | {cpu_ml_mb:>7.1f} | {cpu_other_mb:>7.1f} | {cpu_total_mb:>7.1f} | {gpu_mb:>7.1f} | {summary}"
                )
                
                RESULTS.append({
                    "provider": provider, "device": device, "model": model, 
                    "steps": steps, "clients": clients, "success": success, 
                    "duration": duration,
                    "cpu_solver_mb": cpu_solver_mb,
                    "cpu_ml_mb": cpu_ml_mb,
                    "cpu_other_mb": cpu_other_mb,
                    "cpu_total_mb": cpu_total_mb,
                    "gpu_mb": gpu_mb,
                    "summary": summary
                })
