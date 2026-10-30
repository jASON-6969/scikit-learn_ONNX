"""Record machine, packages, thread controls and process context."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from threadpoolctl import threadpool_info

PACKAGES = ("numpy", "scipy", "scikit-learn", "skl2onnx", "onnx", "onnxruntime", "matplotlib", "threadpoolctl", "joblib")


def cpu_info() -> dict[str, Any]:
    if os.name == "nt":
        try:
            command = (
                "Get-CimInstance Win32_Processor | "
                "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command],
                capture_output=True, text=True, check=True, timeout=20,
            )
            return {"windows_processors": json.loads(completed.stdout)}
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
            pass
    return {"name": platform.processor(), "logical_processors": os.cpu_count()}


def collect_environment() -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu_info(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
        "native_threadpools_during_measurement": threadpool_info(),
        "processor_affinity": "not pinned; OS scheduler controls placement",
        "power_conditions": "not controlled by the program; record plugged-in/power mode manually",
    }

