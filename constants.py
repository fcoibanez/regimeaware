""" "Module constants"""

from enum import Enum, auto

MIN_OBS = 120  # Minimum number of observations required for conditional-distribution estimation
MICROCAP_THRESHOLD = 0.01  # Micro-cap threshold for market coverage filtering
HISTORY_START_DT = "1963-07-31"
HISTORY_END_DT = "2024-12-31"


class DataConstants(Enum):
    """Enumeration for data constants used in the EVaR portfolio optimization."""

    WDIR = "D:/bin/regimeaware"
    WRDS_USERNAME = "fcoibanez"


class SimulationParameters(Enum):
    """Enumeration for simulation constants."""

    TRIALS = 1000
    OOS_PERIODS = 60
    IS_PERIODS = 600
    NUM_STOCKS = 200
    RISK_AVERSION = (1, 5, 10, 25, 50)


class HMMParameters(Enum):
    """Enumeration for HMM parameters."""

    STATES = 3
    MINCOV = 1e-3
    SEED = 13
    RS = 2
    ITER = 1000
    COV = "diag"
    TOL = 1e-1
    IMPLEMENTATION = "scaling"


class Factors(Enum):
    """Enumeration for factor to be used for training."""

    mktrf = auto()
    smb = auto()
    hml = auto()
    rmw = auto()
    cma = auto()
    umd = auto()

## Setup and Configuration
