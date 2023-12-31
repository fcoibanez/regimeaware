"""Global config file"""
from datetime import datetime
import os

fldr = os.path.abspath(os.path.join(os.getcwd(), '..'))
data_fldr = f"{fldr}/data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2022, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 4  # Number of hidden states to use
estimation_freq = "M"
rebalance_freq = "M"
data_freq = "M"
obs_thresh = 120  # Ten years of observations before estimating the RWLS
forecast_horizon = 1  # In months

ols_windows = [36, 60, 120, 999]  # Lookback windows used in the OLS estimation (months)
gamma_iter = [.5, 1]
tev_budget_iter = [.05, .1]

# Hidden Markov Model parameters
hm_cov = "full"
hm_min_covar = 1E-3
hm_iter = 100
hm_algo = "viterbi"
hm_rs = 1234
hm_tol = 1E-2
hm_params_to_estimate = "tmc"
hm_init_params_to_estimate = ""
