# Compatibility shim: warp 1.8+ removed wp.select in favor of wp.where
import warp as wp
if not hasattr(wp, "select") and hasattr(wp, "where"):
    wp.select = wp.where

from .config import Stage3Config
from .system import Stage3System
from . import energy
from . import utils
from . import defs
from .processing import process
