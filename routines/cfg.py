"""Global config file"""
from datetime import datetime
import os
import numpy as np

fldr = r"D:\My Drive\bin\regimeaware"
data_fldr = f"{fldr}/data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2022, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 3  # Number of hidden states to use
estimation_freq = "M"
rebalance_freq = "M"
data_freq = "M"
obs_thresh = 180  # 15 years of observations before estimating the RWLS
forecast_horizon = 1  # In months
mcap_thresh = .975  # Getting rid of micro-caps

ols_windows = [36, 60, 120, 999]  # Lookback windows used in the OLS estimation (months)
gamma_iter = range(10, 110, 10)
tev_budget_iter = np.arange(.01, .105, .005)

# Hidden Markov Model parameters
hm_cov = "full"
hm_min_covar = 1E-3
hm_iter = 100
hm_algo = "viterbi"
hm_rs = 1234
hm_tol = 1E-10
hm_params_to_estimate = "tmc"
hm_init_params_to_estimate = ""
