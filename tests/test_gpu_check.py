"""Tests for gpu_check.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gpu_check import parse_nvidia_smi, check_availability, get_free_gpus


SAMPLE_OUTPUT = """index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%]
0, NVIDIA GeForce RTX 3090, 24576 MiB, 1234 MiB, 5 %
1, NVIDIA GeForce RTX 3090, 24576 MiB, 20000 MiB, 95 %
2, NVIDIA GeForce RTX 3090, 24576 MiB, 500 MiB, 10 %"""


def test_parse_nvidia_smi_basic():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    assert len(gpus) == 3
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 3090"
    assert gpus[0]["memory_total_mib"] == 24576
    assert gpus[0]["memory_used_mib"] == 1234
    assert gpus[0]["utilization_pct"] == 5


def test_parse_nvidia_smi_empty():
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("index, name\n") == []


def test_check_availability():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    gpus = check_availability(gpus, util_threshold=30)
    assert gpus[0]["available"] is True   # 5% < 30%
    assert gpus[1]["available"] is False  # 95% >= 30%
    assert gpus[2]["available"] is True   # 10% < 30%


def test_check_availability_custom_threshold():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    gpus = check_availability(gpus, util_threshold=8)
    assert gpus[0]["available"] is True   # 5% < 8%
    assert gpus[1]["available"] is False  # 95% >= 8%
    assert gpus[2]["available"] is False  # 10% >= 8%


def test_get_free_gpus():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    gpus = check_availability(gpus, util_threshold=30)
    free = get_free_gpus(gpus)
    assert free == [0, 2]


def test_check_availability_memory_threshold():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    # GPU 1 has 20000/24576 MiB used (~81.4%). With memory_threshold=80, it should
    # be unavailable even though util_threshold is generous.
    gpus = check_availability(gpus, util_threshold=100, memory_threshold=80)
    assert gpus[0]["available"] is True   # 5% util, ~5% mem — both under
    assert gpus[1]["available"] is False  # 95% util ok (threshold 100), but ~81% mem >= 80%
    assert gpus[2]["available"] is True   # 10% util, ~2% mem — both under

    # With memory_threshold=90, GPU 1's ~81% memory is fine, so only util matters.
    gpus2 = parse_nvidia_smi(SAMPLE_OUTPUT)
    gpus2 = check_availability(gpus2, util_threshold=100, memory_threshold=90)
    assert gpus2[1]["available"] is True  # 95% util < 100 threshold, ~81% mem < 90%


def test_check_availability_both_thresholds():
    gpus = parse_nvidia_smi(SAMPLE_OUTPUT)
    gpus = check_availability(gpus, util_threshold=30, memory_threshold=80)
    # GPU 0: 5% util < 30, ~5% mem < 80 — available
    assert gpus[0]["available"] is True
    # GPU 1: 95% util >= 30 AND ~81% mem >= 80 — unavailable on both counts
    assert gpus[1]["available"] is False
    # GPU 2: 10% util < 30, ~2% mem < 80 — available
    assert gpus[2]["available"] is True
    # Verify memory_used_pct is populated for visibility
    assert "memory_used_pct" in gpus[0]
    assert "memory_used_pct" in gpus[1]
    assert gpus[1]["memory_used_pct"] > 80


def test_parse_nvidia_smi_malformed_index():
    malformed = """index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%]
N/A, NVIDIA GeForce RTX 3090, ERR MiB, ERR MiB, ERR %
0, NVIDIA GeForce RTX 3090, 24576 MiB, 1234 MiB, 5 %"""
    gpus = parse_nvidia_smi(malformed)
    assert len(gpus) == 2
    # First GPU should have name but no numeric fields due to malformed values
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 3090"
    assert "index" not in gpus[0]
    # Second GPU should parse normally
    assert gpus[1]["index"] == 0
    assert gpus[1]["memory_total_mib"] == 24576
