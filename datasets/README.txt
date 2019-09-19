Cryptocurrency data was obtained from https://www.cryptodatadownload.com/data/northamerican/ using the 3 largest exchanges.

Google Trend data obtained using the pytrendsdaily package at https://github.com/dsgerbc/pytrendsdaily.
    The normal pytrend pseudo-API wasn't enough since daily trend data is required to match the financial data
    and for queries larger than 90 days the pytrend package will only return weekly data.