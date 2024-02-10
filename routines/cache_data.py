if __name__ == "__main__":
    import wrds
    import pandas as pd
    import numpy as np
    from regimeaware.routines import cfg
    from regimeaware.core.utils import assign_industry, assign_exchange
    from tqdm import tqdm

    # ---------------------------------------------------------------------
    # Preamble
    conn = wrds.Connection(wrds_username='fcoibanez')

    # ---------------------------------------------------------------------
    # F&F Factors
    query = \
     f'''
     SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
     FROM ff.fivefactors_monthly
     '''
    ff = conn.raw_sql(query)
    ff['date'] = pd.DatetimeIndex(ff['date']) + pd.offsets.MonthEnd(0)
    ff = ff.set_index('date')
    ff.to_pickle(f'{cfg.data_fldr}/ff.pkl')

    # Daily version
    query = \
     f'''
     SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
     FROM ff.fivefactors_daily
     '''
    ff_daily = conn.raw_sql(query)
    ff_daily['date'] = pd.DatetimeIndex(ff_daily['date'])
    ff_daily = ff_daily.set_index('date')
    ff_daily.to_pickle(f'{cfg.data_fldr}/ff_daily.pkl')

    # ---------------------------------------------------------------------
    # CRSP (borrowed from https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html)
    query = \
        """
        SELECT msf.permno, msf.date, date_trunc('month', msf.date)::date as month, msf.ret, msf.shrout, msf.vol, msf.prc, msf.altprc, msenames.exchcd, msenames.siccd, msedelist.dlret, msedelist.dlstcd
        FROM crsp.msf AS msf
        LEFT JOIN crsp.msenames as msenames ON msf.permno = msenames.permno AND msenames.namedt <= msf.date AND msf.date <= msenames.nameendt
        LEFT JOIN crsp.msedelist as msedelist
        ON msf.permno = msedelist.permno AND date_trunc('month', msf.date)::date = date_trunc('month', msedelist.dlstdt)::date
        WHERE msf.date BETWEEN '01/01/1960' AND '12/31/2022' AND msenames.shrcd IN (10, 11)
        """
    crsp_monthly = conn.raw_sql(query)

    # Next, we follow Bali, Engle, and Murray (2016) in transforming listing exchange codes to explicit exchange names.
    crsp_monthly["exchange"] = crsp_monthly["exchcd"].apply(assign_exchange)
    crsp_monthly["industry"] = crsp_monthly["siccd"].apply(assign_industry)

    """
    The delisting of a security usually results when a company ceases operations, 
    declares bankruptcy, merges, does not meet listing requirements, or seeks to 
    become private.
    """

    conditions_delisting = [
        crsp_monthly["dlstcd"].isna(),
        (~crsp_monthly["dlstcd"].isna()) &
        (~crsp_monthly["dlret"].isna()),
        crsp_monthly["dlstcd"].isin([500, 520, 580, 584]) |
        ((crsp_monthly["dlstcd"] >= 551) &
         (crsp_monthly["dlstcd"] <= 574)),
        crsp_monthly["dlstcd"] == 100
    ]

    choices_delisting = [
        crsp_monthly["ret"],
        crsp_monthly["dlret"],
        -0.30,
        crsp_monthly["ret"]
    ]

    crsp_monthly = crsp_monthly.assign(ret_adj=np.select(conditions_delisting, choices_delisting, default=-1))
    crsp_monthly = crsp_monthly.drop(columns=["dlret", "dlstcd"])
    crsp_monthly['date'] = pd.DatetimeIndex(crsp_monthly['month']) + pd.offsets.MonthEnd(0)
    crsp_monthly = crsp_monthly.set_index(['date', 'permno'])
    crsp_monthly = crsp_monthly.sort_index()
    crsp_monthly = crsp_monthly.join(ff['rf'], on='date')
    crsp_monthly['excess_ret'] = crsp_monthly['ret_adj'] - crsp_monthly['rf']
    crsp_monthly['dollar_vol'] = crsp_monthly['prc'].mul(crsp_monthly['vol'])
    crsp_monthly.drop(['rf', 'month'], axis=1, inplace=True)
    crsp_monthly['mktcap'] = crsp_monthly['shrout'].mul(crsp_monthly['altprc']).abs().replace(0, np.nan)
    crsp_monthly.to_pickle(f'{cfg.data_fldr}/crsp.pkl')

    # ---------------------------------------------------------------------
    # CRSP (daily returns)
    query_template = \
        """
        SELECT dsf.permno, dsf.date, dsf.ret, dsf.shrout, dsf.vol, dsf.prc, msenames.siccd, msenames.naics
        FROM crsp.dsf AS dsf
        LEFT JOIN crsp.msenames AS msenames ON dsf.permno = msenames.permno
        WHERE dsf.date BETWEEN '01/01/{yr}' AND '12/31/{yr}' 
        AND msenames.shrcd IN (10, 11)
        AND msenames.exchcd IN (1, 2, 3, 31, 32, 33)
        """

    collect_responses = []
    for yr in tqdm(range(1963, 2023), desc='Pulling daily responses'):
        query = query_template.format(yr=yr)
        res = conn.raw_sql(query)
        rt = res.set_index(['date', 'permno']).squeeze()
        rt = rt.sort_index()
        rt = rt[~rt.index.duplicated(keep='last')]
        collect_responses += [rt.dropna(subset='ret')]

    crsp_daily = pd.concat(collect_responses, axis=0)
    crsp_daily = crsp_daily.reset_index()
    crsp_daily['date'] = pd.DatetimeIndex(crsp_daily['date'])
    crsp_daily = crsp_daily.set_index(['date', 'permno']).squeeze()
    crsp_daily.to_pickle(f'{cfg.data_fldr}/crsp_daily.pkl')
