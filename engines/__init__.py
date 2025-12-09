"""
Engines package: maps block labels to engine classes.
"""

from .cars_eubo import CarsEUBOEngine
from .cars_bald import CarsBALDEngine
from .gambles_eubo import GamblesEUBOEngine
from .gambles_bald import GamblesBALDEngine


_BLOCK_ORDER = [
    "cars_eubo",
    "cars_bald",
    "gambles_eubo",
    "gambles_bald",
]


def get_block_order() -> list[str]:
    return list(_BLOCK_ORDER)


_ENGINE_MAP = {
    "cars_eubo": CarsEUBOEngine,
    "cars_bald": CarsBALDEngine,
    "gambles_eubo": GamblesEUBOEngine,
    "gambles_bald": GamblesBALDEngine,
}


def get_engine_class(block_label: str):
    return _ENGINE_MAP[block_label]
