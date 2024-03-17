import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, base_dir)
from CNN_IC import myutils
from CNN_IC import configs

__all__ = (
    "configs",
    "myutils",
)