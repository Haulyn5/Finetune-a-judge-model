from __future__ import annotations

import subprocess


def choose_idle_gpu() -> int | None:
    """Pick the visible GPU with the lowest current memory usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    candidates: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        gpu_index_text, memory_used_text = [part.strip() for part in line.split(",", maxsplit=1)]
        candidates.append((int(gpu_index_text), int(memory_used_text)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[0][0]
