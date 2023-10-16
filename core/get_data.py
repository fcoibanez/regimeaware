import wrds
import pandas as pd
from datetime import datetime
import os

# ---------------------------------------------------------------------
# Preamble
conn = wrds.Connection(wrds_username='fcoibanez')
fldr = os.path.abspath(os.path.join(os.getcwd(), '..'))

# ---------------------------------------------------------------------
# S&P 500
query = \
 f'''
 SELECT sp.permno, meta.permco, meta.comnam, meta.namedt, meta.cusip, meta.ticker, sp.start, sp.ending
 FROM crsp_a_indexes.dsp500list AS sp
 LEFT JOIN crsp.stocknames AS meta
 ON sp.permno = meta.permno
 '''
meta_sp = conn.raw_sql(query)
meta_sp.to_pickle(f'{fldr}/data/meta_sp.pkl')

# Historical Returns
id_str = ', '.join([f"'{str(int(x))}'" for x in meta_sp['permno'].unique()])
query = \
 f'''
 SELECT date, permno, ret
 FROM crsp_a_stock.dsf
 WHERE permno IN ({id_str})
 '''
rt_long = conn.raw_sql(query)
rt = pd.pivot_table(rt_long, index='date', columns='permno', values='ret')
rt.columns = rt.columns.astype(int)
rt.index = pd.DatetimeIndex(rt.index)
rt.sort_index(inplace=True)
rt.to_pickle(f'{fldr}/data/rt_sp.pkl')

# Inclusion flags
consti = meta_sp.sort_values('namedt').groupby(['permno', 'start']).last()
consti = consti.reset_index()
consti = consti.sort_values(['permno', 'start'])

idx_y = pd.date_range(start=meta_sp['start'].min(), end=datetime(2022, 12, 31), freq='A')
is_tradable = pd.DataFrame(False, index=idx_y, columns=meta_sp['permno'].unique())

for row in range(consti.shape[0]):
 key = consti.loc[row, 'permno']
 from_dt, till_dt = consti.loc[row, ['start', 'ending']]
 is_tradable.loc[from_dt:till_dt, key] = True

is_tradable_m = is_tradable.resample('BM').ffill()
is_tradable_m = is_tradable_m.shift(1).dropna()

is_tradable_m.to_pickle(f'{fldr}/data/is_tradable_sp.pkl')

# ---------------------------------------------------------------------
# F&F Factors
query = \
 f'''
 SELECT *
 FROM ff.fivefactors_daily
 '''
ff = conn.raw_sql(query)
ff = ff.set_index('date')
ff.index = pd.DatetimeIndex(ff.index)
ff.to_pickle(f'{fldr}/data/ff.pkl')
