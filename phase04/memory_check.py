import torch
from openai import OpenAI
import os
import csv
import time
from typing import Any, List, Dict


def check_hardware():
    """Display CUDA hardware information."""
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"HOW MANY CUDA: {torch.cuda.device_count()}")
    print(f"GPU DEVICE NAME: {torch.cuda.get_device_name(0)}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024**3)
    total_gb = total_mem / (1024**3)
    print(free_gb, total_gb)


if __name__ == "__main__":
    check_hardware()
