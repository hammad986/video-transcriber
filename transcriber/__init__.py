"""
Video Transcriber System 
"""

__version__ = "1.0.0"
__author__ = "Mohd Hammad"

from .config import Config
from .logger import setup_logger

__all__ = [
    "Config",
    "setup_logger", 
]
