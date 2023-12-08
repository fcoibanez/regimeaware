"""Global config file"""
from datetime import datetime
import os

fldr = os.path.abspath(os.path.join(os.getcwd(), '..'))
data_fldr = f"{fldr}/data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2022, 12, 30)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 4  # Number of hidden states to use
estimation_freq = "M"
rebalance_freq = "M"
data_freq = "M"
obs_thresh = 120  # Ten years of observations before estimating the RWLS
forecast_horizon = 3  # In months

# Hidden Markov Model parameters
hm_cov = "full"
hm_min_covar = 1E-3
hm_iter = 100
hm_algo = "viterbi"
hm_rs = 1234
hm_tol = 1E-2
hm_params_to_estimate = "stmc"
hm_init_params_to_estimate = ""
