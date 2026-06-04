import argparse
import os
import subprocess
import sys
import time
import threading
import signal
import re
from pathlib import Path

try:
    import pynvml
except Exception:
    pynvml = None

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

PHYDLL_DL_MODES = ["cpp", "python"]

DEFAULT_GPU_ID = 3
GPU_RANKS_TO_EXCLUDE = int(os.environ.get("GPU_RANKS_TO_EXCLUDE", "1"))

RESULTS = []

class ResourceMonitor:
    def __init__(self, target_gpu=None, run_id=None, run_meta=None, log_dir=None):
        self.target_gpu = target_gpu
        self.run_id = run_id
        self.run_meta = run_meta or {}
        self.log_dir = log_dir
        self.max_cpu_solver_kb = 0
        self.max_cpu_ml_kb = 0
        self.max_cpu_other_kb = 0
        self.max_cpu_total_kb = 0
        self.max_gpu_mem_mb = 0
        self.gpu_proc_max_mb = {}
        self.max_snapshot = []
        self.max_snapshot_total_kb = 0
        self.running = False
        self.root_pid = None
        self.root_pgid = None

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

    def _read_proc_name(self, pid):
        try:
            with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as f:
                return f.read().strip()
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

    def _get_run_pids(self):
        if not self.run_id:
            return []
        try:
            pids = []
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = entry
                env = self._read_proc_env(pid)
                if env.get("MODULE_TEST_RUN_ID") != self.run_id:
                    continue
                if self._is_excluded_gpu_rank(env):
                    continue
                pids.append(pid)
            return pids
        except Exception:
            return []

    def _get_pgid_pids(self):
        if self.root_pgid is None:
            return []
        try:
            cmd = ["ps", "-e", "-o", "pid,pgid", "--no-headers"]
            out = subprocess.check_output(cmd, text=True)
            pids = []
            for line in out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] == str(self.root_pgid):
                    pids.append(parts[0])
            return pids
        except Exception:
            return []

    def get_group_memory(self):
        if not self.root_pid:
            return 0, 0, 0, 0
        try:
            pid_set = set(self._get_run_pids())
            if not pid_set:
                pid_set = set(self._get_pgid_pids())
            if not pid_set:
                pid_set = self._get_tree_pids()
            if not pid_set:
                return 0, 0, 0, 0

            cmd = ["ps", "-e", "-o", "pid,rss", "--no-headers"]
            out = subprocess.check_output(cmd, text=True)
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

    def _get_ppid_map(self):
        ppid_map = {}
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = entry
                try:
                    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                        stat = f.read().split()
                    if len(stat) > 3:
                        ppid_map[pid] = stat[3]
                except Exception:
                    continue
        except Exception:
            return {}
        return ppid_map

    def _get_tree_pids(self):
        if not self.root_pid:
            return set()
        ppid_map = self._get_ppid_map()
        if not ppid_map:
            return set()
        root = str(self.root_pid)
        tree = {root}
        changed = True
        while changed:
            changed = False
            for pid, ppid in ppid_map.items():
                if ppid in tree and pid not in tree:
                    tree.add(pid)
                    changed = True
        return tree

    def _truncate_text(self, text, max_len=30):
        if not text or len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _nvml_gpu_processes(self):
        proc_rows = []
        total_mb = 0
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.target_gpu)
            procs = []
            try:
                procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            except Exception:
                pass
            try:
                procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            except Exception:
                pass

            for proc in procs:
                pid = str(proc.pid)
                used_mb = int(proc.usedGpuMemory / (1024 * 1024))
                name = self._read_proc_name(pid) or self._read_proc_cmdline(pid) or pid
                proc_rows.append((pid, name, used_mb))
                total_mb += used_mb
            return proc_rows, total_mb
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_gpu_processes(self):
        if self.target_gpu is None:
            return [], 0
        try:
            if pynvml is not None:
                proc_rows, total_mb = self._nvml_gpu_processes()
                if proc_rows or total_mb > 0:
                    return proc_rows, total_mb

            cmd = [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.target_gpu),
            ]
            out = subprocess.check_output(cmd, text=True)
            proc_rows = []
            total_mb = 0
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    pid = parts[0]
                    name = parts[1] or self._read_proc_name(pid) or self._read_proc_cmdline(pid) or pid
                    used_mb = int(parts[2])
                    proc_rows.append((pid, name, used_mb))
                    total_mb += used_mb
            if proc_rows or total_mb > 0:
                return proc_rows, total_mb

            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.target_gpu),
            ]
            out = subprocess.check_output(cmd, text=True)
            total_mb = int(out.strip().splitlines()[0]) if out.strip() else 0
            return [], total_mb
        except Exception:
            return [], 0

    def _snapshot_tree(self, pid_set, gpu_by_pid):
        rows = []
        for pid in sorted(pid_set, key=int):
            cmdline = self._read_proc_cmdline(pid)
            name = self._read_proc_name(pid) or pid
            rss_kb = 0
            try:
                with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                    stat = f.read().split()
                rss_pages = int(stat[23]) if len(stat) > 23 else 0
                rss_kb = rss_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
            except Exception:
                rss_kb = 0
            gpu_mb = gpu_by_pid.get(pid, 0)
            rows.append((pid, name, rss_kb, gpu_mb, cmdline))
        return rows

    def _format_tree(self, rows):
        rows_by_pid = {pid: (name, rss_kb, gpu_mb, cmdline) for pid, name, rss_kb, gpu_mb, cmdline in rows}
        ppid_map = self._get_ppid_map()
        children = {}
        for pid in rows_by_pid:
            ppid = ppid_map.get(pid)
            if not ppid or ppid not in rows_by_pid:
                continue
            children.setdefault(ppid, []).append(pid)
        for pid in children:
            children[pid].sort(key=int)

        root = str(self.root_pid) if self.root_pid else None
        if not root or root not in rows_by_pid:
            roots = sorted(
                [pid for pid in rows_by_pid if ppid_map.get(pid) not in rows_by_pid],
                key=int,
            )
        else:
            roots = [root]

        lines = []

        def add_node(pid, prefix, is_last):
            name, rss_kb, gpu_mb, cmdline = rows_by_pid[pid]
            short_cmd = self._truncate_text(cmdline, 30)
            branch = "└── " if is_last else "├── "
            label = f"{pid} {name} [{rss_kb:>7} KB, {gpu_mb:>4} MB] {short_cmd}"
            lines.append(prefix + (branch if prefix else "") + label)

            kids = children.get(pid, [])
            if not kids:
                return
            next_prefix = prefix + ("    " if is_last else "│   ")
            for idx, child_pid in enumerate(kids):
                add_node(child_pid, next_prefix, idx == len(kids) - 1)

        for idx, pid in enumerate(roots):
            add_node(pid, "", idx == len(roots) - 1)

        return "\n".join(lines)

    def _write_snapshot(self, rows):
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            name_parts = [
                self.run_meta.get("provider", "unknown"),
                self.run_meta.get("dl_mode", "-"),
                self.run_meta.get("device", "unknown"),
                self.run_meta.get("model", "unknown"),
                f"{self.run_meta.get('steps', '0')}_{self.run_meta.get('clients', '0')}",
                self.run_id or "run",
            ]
            safe_name = "_".join(name_parts).replace("/", "-")
            out_path = Path(self.log_dir) / f"memtree_{safe_name}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"root_pid: {self.root_pid}\n")
                f.write(f"run_id: {self.run_id}\n")
                f.write(self._format_tree(rows))
                f.write("\n\n\n")
                f.write("gpu_processes (pid\tname\tmax_gpu_mb)\n")
                if self.gpu_proc_max_mb:
                    for pid, (name, used_mb) in sorted(
                        self.gpu_proc_max_mb.items(),
                        key=lambda item: (-item[1][1], item[1][0]),
                    ):
                        f.write(f"{pid}\t{name}\t{used_mb}\n")
                else:
                    f.write("(none)\n")
        except Exception:
            pass

    def monitor(self):
        while self.running:
            pid_set = set(self._get_run_pids())
            if not pid_set:
                pid_set = set(self._get_pgid_pids())
            if not pid_set:
                pid_set = self._get_tree_pids()
            tree_pids = self._get_tree_pids()
            pgid_pids = set(self._get_pgid_pids())
            snapshot_pids = set(tree_pids) | pgid_pids
            solver_rss, ml_rss, other_rss, total_rss = self.get_group_memory()
            gpu_rows, gpu_mem = self.get_gpu_processes()
            gpu_by_pid = {pid: used_mb for pid, _name, used_mb in gpu_rows}
            
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
            for pid, name, used_mb in gpu_rows:
                prev = self.gpu_proc_max_mb.get(pid)
                if prev is None or used_mb > prev[1]:
                    self.gpu_proc_max_mb[pid] = (name, used_mb)
            if snapshot_pids and total_rss > self.max_snapshot_total_kb:
                self.max_snapshot_total_kb = total_rss
                self.max_snapshot = self._snapshot_tree(snapshot_pids, gpu_by_pid)
            
            time.sleep(0.5) # Polling interval

    def start(self, root_pid):
        self.root_pid = root_pid
        try:
            self.root_pgid = os.getpgid(root_pid)
        except Exception:
            self.root_pgid = None
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
        if self.max_snapshot:
            self._write_snapshot(self.max_snapshot)

def run_command(cmd, env, target_gpu=None, run_meta=None, log_dir=None):
    run_id = f"{int(time.time() * 1000)}_{os.getpid()}"
    env["MODULE_TEST_RUN_ID"] = run_id
    monitor = ResourceMonitor(target_gpu, run_id, run_meta=run_meta, log_dir=log_dir)
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
            monitor.gpu_proc_max_mb,
            result_summary,
            output,
        )
    except Exception as e:
        return False, 0, 0, 0, 0, 0, 0, {}, "ERROR", str(e)

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

parser = argparse.ArgumentParser(description="Run module_test matrix and write a markdown table.")
parser.add_argument("--out", dest="out_path", default="", help="Optional output file path for the table")
parser.add_argument("--log-dir", dest="log_dir", default="", help="Optional directory for memory tree logs")
args = parser.parse_args()

out_f = open(args.out_path, "w", encoding="utf-8") if args.out_path else None

def emit(line):
    print(line, flush=True)
    if out_f:
        out_f.write(line + "\n")
        out_f.flush()

def emit_progress(done, total, start_ts):
    if done <= 0:
        return
    elapsed = time.time() - start_ts
    rate = elapsed / done
    eta = rate * (total - done)
    msg = f"\rProgress: {done}/{total} | Elapsed: {elapsed:>6.1f}s | ETA: {eta:>6.1f}s"
    sys.stdout.write(msg)
    sys.stdout.flush()

# Header
emit(f"| {'Provider':<9} | {'DL':<6} | {'Dev':<4} | {'Model':<11} | {'St/Cl':<5} | {'Stat':<2} | {'Time':<6} | {'CPU_S':<7} | {'CPU_M':<7} | {'CPU_O':<7} | {'CPU_T':<7} | {'GPU(MB)':<7} | {'GPU_Procs'} | {'Results'}")
emit(f"|{'-'*11}|{'-'*8}|{'-'*6}|{'-'*13}|{'-'*7}|{'-'*6}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*12}|{'-'*40}")

ss_port = 7200
total_tests = 0
for provider in PROVIDERS:
    dl_count = len(PHYDLL_DL_MODES) if provider == "PHYDLL" else 1
    total_tests += dl_count * len(DEVICES) * len(MODELS) * len(WORKLOADS)
done_tests = 0
start_ts = time.time()

for provider in PROVIDERS:
    for device in DEVICES:
        for model in MODELS:
            update_toml(provider, device, model)
            dl_modes = PHYDLL_DL_MODES if provider == "PHYDLL" else ["-"]
            for dl_mode in dl_modes:
                for steps, clients in WORKLOADS:
                    env = os.environ.copy()
                    env["PROVIDER"] = provider
                    env["DEVICE"] = device
                    env["STEPS"] = str(steps)
                    env["CLIENTS"] = str(clients)
                    env["COMPILE"] = "0"
                    env["MODEL"] = model
                    if provider == "PHYDLL":
                        env["USE_PYTHON_DL_CLIENT"] = "1" if dl_mode == "python" else "0"
                    
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

                    run_meta = {
                        "provider": provider,
                        "dl_mode": dl_mode,
                        "device": device,
                        "model": model,
                        "steps": steps,
                        "clients": clients,
                    }
                    success, duration, cpu_solver_mb, cpu_ml_mb, cpu_other_mb, cpu_total_mb, gpu_mb, gpu_procs, summary, full_log = run_command(
                        ["./run.sh"],
                        env,
                        target_gpu,
                        run_meta=run_meta,
                        log_dir=args.log_dir or None,
                    )
                    if gpu_procs:
                        sorted_procs = sorted(gpu_procs.items(), key=lambda item: (-item[1][1], item[1][0]))
                        gpu_procs_str = "; ".join(
                            f"{pid} {name}: {used}MiB" for pid, (name, used) in sorted_procs
                        )
                    else:
                        gpu_procs_str = "-"
                    
                    status = "✅" if success else "❌"
                    emit(
                        f"| {provider:<9} | {dl_mode:<6} | {device:<4} | {model:<11} | {steps}/{clients:<3} | {status:<2} | {duration:>5.1f}s | "
                        f"{cpu_solver_mb:>7.1f} | {cpu_ml_mb:>7.1f} | {cpu_other_mb:>7.1f} | {cpu_total_mb:>7.1f} | {gpu_mb:>7.1f} | {gpu_procs_str} | {summary}"
                    )
                    
                    RESULTS.append({
                        "provider": provider,
                        "dl_mode": dl_mode,
                        "device": device,
                        "model": model,
                        "steps": steps,
                        "clients": clients,
                        "success": success,
                        "duration": duration,
                        "cpu_solver_mb": cpu_solver_mb,
                        "cpu_ml_mb": cpu_ml_mb,
                        "cpu_other_mb": cpu_other_mb,
                        "cpu_total_mb": cpu_total_mb,
                        "gpu_mb": gpu_mb,
                        "gpu_procs": gpu_procs_str,
                        "summary": summary
                    })

                    done_tests += 1
                    emit_progress(done_tests, total_tests, start_ts)

if done_tests:
    sys.stdout.write("\n")
    sys.stdout.flush()

if out_f:
    out_f.close()
