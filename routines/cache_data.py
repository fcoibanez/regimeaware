if __name__ == "__main__":
    import wrds
    import pandas as pd
    import os
    import numpy as np

    # ---------------------------------------------------------------------
    # Preamble
    conn = wrds.Connection(wrds_username='fcoibanez')
    fldr = os.path.abspath(os.path.join(os.getcwd(), '..'))

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
    ff.to_pickle(f'{fldr}/data/ff.pkl')

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
    def assign_exchange(exchcd):
        if exchcd in [1, 31]:
            return "NYSE"
        elif exchcd in [2, 32]:
            return "AMEX"
        elif exchcd in [3, 33]:
            return "NASDAQ"
        else:
            return "Other"

    crsp_monthly["exchange"] = crsp_monthly["exchcd"].apply(assign_exchange)

    # transform industry codes to industry descriptions
    def assign_industry(siccd):
        if 1 <= siccd <= 999:
            return "Agriculture"
        elif 1000 <= siccd <= 1499:
            return "Mining"
        elif 1500 <= siccd <= 1799:
            return "Construction"
        elif 2000 <= siccd <= 3999:
            return "Manufacturing"
        elif 4000 <= siccd <= 4899:
            return "Transportation"
        elif 4900 <= siccd <= 4999:
            return "Utilities"
        elif 5000 <= siccd <= 5199:
            return "Wholesale"
        elif 5200 <= siccd <= 5999:
            return "Retail"
        elif 6000 <= siccd <= 6799:
            return "Finance"
        elif 7000 <= siccd <= 8999:
            return "Services"
        elif 9000 <= siccd <= 9999:
            return "Public"
        else:
            return "Missing"


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
    crsp_monthly.to_pickle(f'{fldr}/data/crsp.pkl')
