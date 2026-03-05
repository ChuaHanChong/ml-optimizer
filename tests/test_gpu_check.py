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
