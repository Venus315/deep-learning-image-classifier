import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

__all__ = [
    "compute_dataset_normalization",
    "apply_weight_initialization",
    "progress_bar"
]

def compute_dataset_normalization(dataset, num_workers=2):
    """Calculate per-channel mean and std deviation for normalization."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("[Info] Computing dataset normalization stats...")
    for img, _ in loader:
        for c in range(3):
            mean[c] += img[:, c].mean()
            std[c] += img[:, c].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def apply_weight_initialization(model):
    """Initialize model weights using Kaiming and normal strategies."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

_TERM_WIDTH = shutil.get_terminal_size((80, 20)).columns
_BAR_LENGTH = 65
_last_time = time.time()
_start_time = _last_time

def progress_bar(current, total, msg=None):
    """Visual progress bar with timing and optional message."""
    global _last_time, _start_time
    if current == 0:
        _start_time = time.time()

    cur_len = int(_BAR_LENGTH * current / total)
    rest_len = int(_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - _last_time
    tot_time = cur_time - _start_time
    _last_time = cur_time

    message_parts = [
        f" Step: {format_time(step_time)}",
        f" | Total: {format_time(tot_time)}"
    ]
    if msg:
        message_parts.append(f" | {msg}")
    msg_str = ''.join(message_parts)
    sys.stdout.write(msg_str)

    pad = _TERM_WIDTH - _BAR_LENGTH - len(msg_str) - 3
    sys.stdout.write(' ' * pad)
    sys.stdout.write('\b' * (_TERM_WIDTH - int(_BAR_LENGTH / 2) + 2))
    sys.stdout.write(f' {current + 1}/{total} ')

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    """Format seconds into readable duration string."""
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    seconds_int = int(seconds)
    millis = int((seconds - seconds_int) * 1000)

    output = []
    count = 0
    for unit, val in zip(['D', 'h', 'm', 's', 'ms'], [days, hours, minutes, seconds_int, millis]):
        if val > 0 and count < 2:
            output.append(f"{val}{unit}")
            count += 1
    return ''.join(output) if output else '0ms'
