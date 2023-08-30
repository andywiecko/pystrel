"""
    .. include:: ../README.md
"""
from . import combinadics
from . import terms
from . import operators
from . import spectrum
from .dynamics import propagate
from .laboratory import measure, project
from .model import Model
from .parameters import Parameters

__version__ = "0.1.0"
