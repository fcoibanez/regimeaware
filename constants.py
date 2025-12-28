""""Module constants"""
from enum import Enum

class DataConstants(Enum):
    """Enumeration for data constants used in the EVaR portfolio optimization."""

    WDIR = "D:/bin/regimeaware"
    WRDS_USERNAME = "fcoibanez"

class SimulationConstants(Enum):
    """Enumeration for simulation constants."""

    N_SIMULATIONS = 1000
    TIME_HORIZON = 252  # Number of trading days in a year
