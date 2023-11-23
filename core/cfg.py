"""Global config file"""
from datetime import datetime
import os

fldr = os.path.abspath(os.path.join(os.getcwd(), '..'))
data_fldr = f"{fldr}/data"

trn_start_dt = datetime(1926, 12, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2022, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study
n_states = 4  # Number of hidden states to use
rebalance_freqstr = "BM"

# Hidden Markov Model parameters
hm_cov = 'full'
hm_min_covar = 1E-4
hm_iter = 100
hm_algo = 'viterbi'
hm_rs = 0
hm_tol = 1E2
