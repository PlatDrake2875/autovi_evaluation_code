#!/usr/bin/env python3
"""Parallel experiment runner with worker pool and live status."""

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict

CATEGORIES = [
    "engine_wiring", "pipe_clip", "pipe_staple",
    "tank_screw", "underbody_pipes", "underbody_screw"
]

# Global status tracking
status_lock = threading.Lock()
status: Dict[str, str] = {}
start_times: Dict[str, float] = {}

@dataclass
class Experiment:
    name: str
    command: List[str]

def build_experiments() -> List[Experiment]:
    """Build list of all experiments to run."""
    experiments = []
    uv = "/home/adrian/.local/bin/uv"

    # 1. Centralized baseline (one per category)
    for cat in CATEGORIES:
        experiments.append(Experiment(
            name=f"central_{cat[:6]}",
            command=[
                uv, "run", "python", "scripts/train_centralized.py",
                "--config", "experiments/configs/baseline/patchcore_config.yaml",
                "--data_dir", "dataset",
                "--output_dir", f"outputs/centralized/{cat}",
                "--objects", cat,
            ]
        ))

    # 2-4. Federated + DP (eps=1.0, 5.0, 10.0)
    for eps in [1.0, 5.0, 10.0]:
        for cat in CATEGORIES:
            experiments.append(Experiment(
                name=f"dp{int(eps)}_{cat[:6]}",
                command=[
                    uv, "run", "python", "scripts/train_federated.py",
                    "--config", "experiments/configs/federated/fedavg_dp_config.yaml",
                    "--data_root", "dataset",
                    "--dp_epsilon", str(eps),
                    "--categories", cat,
                    "--output_dir", f"outputs/federated/dp_eps{int(eps)}/{cat}",
                ]
            ))

    # 5. Federated + Robust (clean)
    for cat in CATEGORIES:
        experiments.append(Experiment(
            name=f"robust_{cat[:6]}",
            command=[
                uv, "run", "python", "scripts/train_federated.py",
                "--config", "experiments/configs/federated/fedavg_iid_config.yaml",
                "--data_root", "dataset",
                "--robust_aggregation", "coordinate_median",
                "--categories", cat,
                "--output_dir", f"outputs/federated/robust_clean/{cat}",
            ]
        ))

    # 6. Federated + Robust (attack)
    for cat in CATEGORIES:
        experiments.append(Experiment(
            name=f"attack_{cat[:6]}",
            command=[
                uv, "run", "python", "scripts/train_federated.py",
                "--config", "experiments/configs/federated/fedavg_iid_config.yaml",
                "--data_root", "dataset",
                "--robust_aggregation", "coordinate_median",
                "--simulate_attack", "scaling",
                "--malicious_fraction", "0.2",
                "--categories", cat,
                "--output_dir", f"outputs/federated/robust_attack/{cat}",
            ]
        ))

    return experiments

def print_status(total: int):
    """Print current status of all experiments."""
    with status_lock:
        running = [k for k, v in status.items() if v == "RUNNING"]
        done = [k for k, v in status.items() if v == "DONE"]
        failed = [k for k, v in status.items() if v == "FAILED"]
        pending = total - len(status)

    now = time.time()
    print(f"\n{'='*70}")
    print(f"STATUS: {len(done)} done | {len(running)} running | {len(failed)} failed | {pending} pending")
    print(f"{'='*70}")

    if running:
        print("Running:")
        for name in running[:10]:
            elapsed = int(now - start_times.get(name, now))
            print(f"  [{elapsed:3d}s] {name}")

    if done:
        print(f"Completed: {', '.join(done[-10:])}" + ("..." if len(done) > 10 else ""))

    if failed:
        print(f"Failed: {', '.join(failed)}")
    print()

def status_monitor(total: int, stop_event: threading.Event):
    """Background thread to print status updates."""
    while not stop_event.is_set():
        print_status(total)
        stop_event.wait(15)  # Update every 15 seconds

def run_experiment(exp: Experiment) -> tuple:
    """Run a single experiment and return result."""
    with status_lock:
        status[exp.name] = "RUNNING"
        start_times[exp.name] = time.time()

    try:
        result = subprocess.run(
            exp.command,
            cwd="/home/adrian/autovi_evaluation_code",
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )
        elapsed = int(time.time() - start_times[exp.name])

        if result.returncode == 0:
            with status_lock:
                status[exp.name] = "DONE"
            print(f"[DONE {elapsed:3d}s] {exp.name}")
            return (exp.name, True, None)
        else:
            with status_lock:
                status[exp.name] = "FAILED"
            err = result.stderr[-300:] if result.stderr else "Unknown"
            print(f"[FAIL {elapsed:3d}s] {exp.name}")
            return (exp.name, False, err)

    except subprocess.TimeoutExpired:
        with status_lock:
            status[exp.name] = "FAILED"
        print(f"[TIMEOUT] {exp.name}")
        return (exp.name, False, "Timeout")
    except Exception as e:
        with status_lock:
            status[exp.name] = "FAILED"
        print(f"[ERROR] {exp.name}: {e}")
        return (exp.name, False, str(e))

def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    experiments = build_experiments()

    print(f"{'='*70}")
    print(f"PARALLEL EXPERIMENT RUNNER")
    print(f"Total: {len(experiments)} experiments | Workers: {max_workers}")
    print(f"{'='*70}")

    # Start status monitor thread
    stop_event = threading.Event()
    monitor = threading.Thread(target=status_monitor, args=(len(experiments), stop_event))
    monitor.daemon = True
    monitor.start()

    results = {"success": [], "failed": []}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in experiments}
        for future in as_completed(futures):
            name, success, error = future.result()
            if success:
                results["success"].append(name)
            else:
                results["failed"].append((name, error))

    stop_event.set()
    total_time = int(time.time() - start_time)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Succeeded: {len(results['success'])}/{len(experiments)}")
    print(f"Total time: {total_time//60}m {total_time%60}s")

    if results["failed"]:
        print(f"\nFailed experiments:")
        for name, error in results["failed"]:
            print(f"  - {name}")
            if error:
                print(f"    {error[:200]}")

    return 0 if not results["failed"] else 1

if __name__ == "__main__":
    main()
