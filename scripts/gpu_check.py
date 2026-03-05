#!/usr/bin/env python3
"""Check GPU availability via nvidia-smi."""

import subprocess
import json
import sys


def parse_nvidia_smi(output: str) -> list[dict]:
    """Parse nvidia-smi CSV output into GPU info dicts."""
    lines = output.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split(",")]
    gpus = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(headers):
            continue
        gpu = {}
        for header, value in zip(headers, values):
            key = header.lower().replace(" ", "_").replace(".", "_")
            try:
                if "index" in key:
                    gpu["index"] = int(value)
                elif "name" in key:
                    gpu["name"] = value
                elif "memory_total" in key or "memory.total" in header.lower():
                    gpu["memory_total_mib"] = int(value.replace("MiB", "").strip())
                elif "memory_used" in key or "memory.used" in header.lower():
                    gpu["memory_used_mib"] = int(value.replace("MiB", "").strip())
                elif "utilization" in key:
                    gpu["utilization_pct"] = int(value.replace("%", "").strip())
            except ValueError:
                continue
        gpus.append(gpu)
    return gpus


def check_availability(gpus: list[dict], util_threshold: int = 30) -> list[dict]:
    """Mark GPUs as available/busy based on utilization threshold."""
    for gpu in gpus:
        util = gpu.get("utilization_pct", 100)
        gpu["available"] = util < util_threshold
    return gpus


def get_free_gpus(gpus: list[dict]) -> list[int]:
    """Return indices of available GPUs."""
    return [g["index"] for g in gpus if g.get("available", False)]


def run(util_threshold: int = 30) -> dict:
    """Run nvidia-smi and return GPU status."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi failed: {result.stderr}", "gpus": []}
    except FileNotFoundError:
        return {"error": "nvidia-smi not found", "gpus": []}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out", "gpus": []}

    gpus = parse_nvidia_smi(result.stdout)
    gpus = check_availability(gpus, util_threshold)
    free = get_free_gpus(gpus)
    return {"gpus": gpus, "free_gpu_indices": free, "num_free": len(free)}


if __name__ == "__main__":
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(json.dumps(run(threshold), indent=2))
