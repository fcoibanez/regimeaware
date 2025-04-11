"""Global config file"""
from datetime import datetime
import os
import numpy as np

# fldr = r"D:\My Drive\bin\regimeaware"
fldr = r"C:\Users\franc\My Drive\bin\regimeaware"
data_fldr = f"{fldr}\data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(1993, 12, 31)  # Backtest start
bt_end_dt = datetime(2023, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 3  # Number of hidden states to use
estimation_freq = "M"
rebalance_freq = "M"
data_freq = "M"
obs_thresh = 240  # 20 years of observations before estimating the RWLS
forecast_horizon = 1  # In months
mcap_thresh = .975  # Getting rid of micro-caps

gamma_iter = range(10, 110, 10)
tev_budget_iter = np.arange(.01, .105, .005)

# Hidden Markov Model parameters
hm_cov = "diag"
hm_min_covar = 1E-3
hm_iter = 1000
hm_algo = "viterbi"
hm_rs = 123
hm_tol = 1E-1
hm_params_to_estimate = "tmc"
hm_init_params_to_estimate = ""
hm_implementation = "scaling" 

# Montecarlo simulations
num_simulations = 1000
# num_stocks = [10, 50, 100, 250, 500, 750, 1000]  # Number of stocks to simulate
num_stocks = [500]  # Number of stocks to simulate
var_alphas = [0.05]  # VaR alphas to use
# var_alphas = [0.01, 0.05, 0.1]  # VaR alphas to use
trn_periods = 600
test_periods = 120