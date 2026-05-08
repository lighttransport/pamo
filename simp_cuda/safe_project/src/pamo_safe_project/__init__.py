# Compatibility shim across Warp versions. Newer Warp exposes wp.where;
# older PaMO-tested Warp exposes wp.select with the same cond/true/false shape.
import warp as wp
if not hasattr(wp, "where") and hasattr(wp, "select"):
    wp.where = wp.select
if not hasattr(wp, "select") and hasattr(wp, "where"):
    wp.select = wp.where

from .config import Stage3Config
from .system import Stage3System
from . import energy
from . import utils
from . import defs
from .processing import process
