import math
import random
from datetime import datetime
import inspect
from pathlib import Path

import numpy as np
import torch

def interval(step=None, itv=None, last_step=None):
    if step is None  or  itv is None:
        return True
    else:
        return (step - 1) % itv >= itv - 1  or  step == last_step

def get_source(path=None, with_name=False):
    if path is None:
        src_fname = inspect.stack()[1].filename
        path = Path(src_fname)
    src_name = path.name
    src_text = path.read_text(encoding='utf-8')
    
    if with_name:
        return src_text, src_name
    else:
        return src_text

def torch_fix_seed(seed=42):
    random.seed(seed) # Python random
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # Pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # cuda
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def format_duration(duration_s, style=0) -> str:
    """
    Converts a duration in seconds to a human-readable format.

    Args:
        duration_s (float): Duration in seconds to be formatted.
        style (int, optional): 
            Formatting style. 
            - 0: Single highest unit (e.g., "1.5ms")
            - 1: Two-level breakdown (e.g., "1m23s")
            Defaults to 0.

    Returns:
        str: Formatted duration string.
    """
    duration_ns = duration_s * 1e9
    units = [("ns", 1), ("μs", 1000), ("ms", 1000), ("s", 1000), ("m", 60), ("h", 60), ("d", 24), ("w", 7), ("x", 10000)]
    # units = [("ns", 1), ("μs", 1000), ("ms", 1000), ("s", 1000), ("m", 60), ("h", 60), ("d", 24), ("y", 365), ("c", 100), ("e", 10)]

    # 最大単位を整数部にしたfloatに変換
    if style == 0:
        unit_b = ""
        t_b = duration_ns
        for unit, divisor in units:
            if t_b < divisor:
                return f"{t_b:.1f}{unit_b}"
            t_b /= divisor
            unit_b = unit
        
    # 最大単位から2段階表示
    elif style == 1:
        def digit_count(n):
            if n == 0:
                return 1
            return int(math.log10(abs(n))) + 1

        quo = int(duration_ns)
        fmt_d = {}
        unit_b = ""
        for unit, divisor in units:
            quo, rem = divmod(quo, divisor)
            fmt_d[unit_b] = (rem, digit_count(divisor - 1))
            unit_b = unit

        fmt_str = ""
        part_count = 0
        for k, v in reversed(fmt_d.items()):
            value, digit = v
            if value > 0:
                if part_count == 0  or  part_count == 1:
                    fmt_str += f"{value:{digit}d}{k}"
                    part_count += 1
                else:
                    break
                    
        return fmt_str
    
def format_time(time, style=0):
    if style == 0:
        return datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M")
    elif style == 1:
        return datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S")
    else:
        raise ValueError(f"Unknown format style: {style}")