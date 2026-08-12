# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable terminal TAO lifecycle status writes.

TAO Core's shared logger remains responsible for progress events. Terminal
events are appended directly so a successful worker cannot leave a final
RUNNING record because of logger state inherited by the distributed worker.
"""

import json
import os
from datetime import datetime


_TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILURE"})


def append_terminal_status(filename: str, status: str, message: str) -> None:
    """Append and verify one terminal TAO JSONL record."""
    if status not in _TERMINAL_STATUSES:
        raise ValueError(f"terminal TAO status must be one of {_TERMINAL_STATUSES}, got {status!r}")

    now = datetime.now()
    record = {
        "date": f"{now.month}/{now.day}/{now.year}",
        "time": f"{now.hour}:{now.minute}:{now.second}",
        "status": status,
        "verbosity": "INFO" if status == "SUCCESS" else "ERROR",
        "message": message,
    }
    os.makedirs(os.path.dirname(os.path.realpath(filename)), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    with open(filename, encoding="utf-8") as stream:
        final_line = next(line for line in reversed(stream.readlines()) if line.strip())
    final_record = json.loads(final_line)
    if final_record.get("status") != status:
        raise RuntimeError(
            f"terminal TAO status verification failed: expected {status}, "
            f"found {final_record.get('status')!r}"
        )
