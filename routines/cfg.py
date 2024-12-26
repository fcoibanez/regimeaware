"""Global config file"""
from datetime import datetime
import os
import numpy as np

fldr = os.path.abspath(os.getcwd())
data_fldr = f"{fldr}/data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2022, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 4  # Number of hidden states to use
estimation_freq = "W-Wed"
rebalance_freq = "M"
data_freq = "W-Wed"
obs_thresh = 260  # Five years of observations before estimating the RWLS
forecast_horizon = 4  # In months

ols_windows = [36, 60, 120, 999]  # Lookback windows used in the OLS estimation (months)
gamma_iter = range(10, 110, 10)
tev_budget_iter = np.arange(.01, .105, .005)

# Hidden Markov Model parameters
hm_cov = "full"
hm_min_covar = 1E-3
hm_iter = 100
hm_algo = "viterbi"
hm_rs = 1234
hm_tol = 1E-2
hm_params_to_estimate = "tmc"
hm_init_params_to_estimate = ""
