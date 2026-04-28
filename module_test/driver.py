#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

from smartsim.experiment import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Start local SmartSim DB and export SSDB endpoint.")
    parser.add_argument("--endpoint-file", default=".ssdb_endpoint", help="File to write host:port")
    parser.add_argument("--done-file", default=".solver_done", help="File that signals solver completion")
    parser.add_argument("--port", type=int, default=6780, help="Database port")
    parser.add_argument("--interface", default="lo", help="Network interface")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Readiness timeout")
    args = parser.parse_args()

    endpoint_file = Path(args.endpoint_file)
    done_file = Path(args.done_file)

    if endpoint_file.exists():
        endpoint_file.unlink()
    if done_file.exists():
        done_file.unlink()

    exp_name = f"module_test_{int(time.time())}"

    exp = Experiment(name=exp_name, launcher="local")
    db = exp.create_database(port=args.port, interface=args.interface, db_nodes=1, single_cmd=False, batch=False)

    exp.start(db, block=False, summary=True)

    start = time.time()
    addresses = None
    while time.time() - start < args.timeout_s:
        try:
            addresses = db.get_address()
            if addresses:
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not addresses:
        exp.stop(db)
        raise RuntimeError("SmartSim database did not become ready in time")

    endpoint = ",".join(addresses)
    endpoint_file.write_text(endpoint + "\n", encoding="utf-8")
    print(f"Database ready. SSDB={endpoint}", flush=True)

    while not done_file.exists():
        time.sleep(0.5)

    exp.stop(db)
    
    print("Solver done. Clean up experiment.", flush=True)
  
    if (os.path.exists(exp_name)):
        os.rmdir(exp_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
