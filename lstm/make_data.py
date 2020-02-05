from os.path import join, dirname, abspath
from os import makedirs
from datetime import date, timedelta, datetime
import re
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time

from lstm.helper import get_project_path


def check_date(date):
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def get_btc_price(from_date, to_date=date.today().strftime("%Y-%m-%d")):
    """
    Retrieves historical bitcoin price data from coinmarketcap.com
    from from_date until present and returns a Pandas DataFrame

    Arguments
    from_date: String of form 'yyyy-mm-dd'
    to_date: String of form 'yyyy-mm-dd'

    Returns:
    pandas DataFrame
    """

    # cryptory produced empty dataframes when using it for this purpose
    # so it is being done manually here
    check_date(from_date)

    price_df = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start={}&end={}".format(
        from_date.replace("-", ""), to_date.replace("-", "")))[2]
    price_df = price_df.assign(Date=pd.to_datetime(price_df['Date']))

    for col in price_df.columns:
        if price_df[col].dtype == np.dtype('O'):
            price_df.loc[price_df[col] == "-", col] = 0
            price_df[col] = price_df[col].astype('int64')
    price_df.columns = [re.sub(r"[^a-z]", "", col.lower())
                        for col in price_df.columns]

    return price_df


def get_btc_trend(from_date, to_date=date.today().strftime("%Y-%m-%d"), kw_list=['bitcoin'], trdays=250, overlap=100,
                  cat=0, geo='', tz=360, gprop='', hl='en-US',
                  sleeptime=1, isPartial_col=False,
                  from_start=False, scale_cols=True):
    """Retrieve daily google trends data for a list of search terms

    Parameters
    ----------
    kw_list : list of search terms (max 5)- see pyTrends for more details
    trdays : the number of days to pull data for in a search
        (the max is around 270, though the website seems to indicate 90)
    overlap : the number of overlapped days when stitching two searches together
    cat : category to narrow results - see pyTrends for more details
    geo : two letter country abbreviation (e.g 'US', 'UK') 
        default is '', which returns global results - see pyTrends for more details
    tz : timezone offset
        (default is 360, which corresponds to US CST - see pyTrends for more details)
    grop : filter results to specific google property
        available options are 'images', 'news', 'youtube' or 'froogle'
        default is '', which refers to web searches - see pyTrends for more details
    hl : language (e.g. 'en-US' (default), 'es') - see pyTrends for more details
    sleeptime : when stiching multiple searches, this sets the period between each
    isPartial_col : remove the isPartial column 
        (default is True i.e. column is removed)
    from_start : when stitching multiple results, this determines whether searches
        are combined going forward or backwards in time
        (default is False, meaning searches are stitched with the most recent first)
    scale_cols : google trend searches traditionally returns scores between 0 and 100
        stitching could produce values greater than 100
        by setting this to True (default), the values will range between 0 and 100

    Returns
    -------
    pandas Dataframe

    Notes
    -----
    This method is essentially a highly restricted wrapper for the pytrends package
    Any issues/questions related to its use would probably be more likely resolved
    by consulting the pytrends github page
    https://github.com/GeneralMills/pytrends
    """

    # heck_date(from_date)
    if len(kw_list) > 5 or len(kw_list) == 0:
        raise ValueError("The keyword list can contain at most 5 words")
    if trdays > 270:
        raise ValueError("trdays must not exceed 270")
    if overlap >= trdays:
        raise ValueError("Overlap can't exceed search days")
    stich_overlap = trdays - overlap
    from_datetime = datetime.strptime(from_date, '%Y-%m-%d')
    to_datetime = datetime.strptime(to_date, '%Y-%m-%d')
    n_days = (to_datetime - from_datetime).days
    # launch pytrends request
    _pytrends = TrendReq(hl=hl, tz=tz)
    # get the dates for each search
    if n_days <= trdays:
        trend_dates = [' '.join([from_date, to_date])]
    else:
        trend_dates = ['{} {}'.format(
            (to_datetime - timedelta(i+trdays)).strftime("%Y-%m-%d"),
            (to_datetime - timedelta(i)).strftime("%Y-%m-%d"))
            for i in range(0, n_days-trdays+stich_overlap,
                           stich_overlap)]
    if from_start:
        trend_dates = trend_dates[::-1]
    try:
        _pytrends.build_payload(kw_list, cat=cat, timeframe=trend_dates[0],
                                geo=geo, gprop=gprop)
    except:
        raise
    output = _pytrends.interest_over_time().reset_index()
    if len(output) == 0:
        raise ValueError('search term returned no results (insufficient data)')
    for date in trend_dates[1:]:
        time.sleep(sleeptime)
        try:
            _pytrends.build_payload(kw_list, cat=cat, timeframe=date,
                                    geo=geo, gprop=gprop)
        except:
            raise
        temp_trend = _pytrends.interest_over_time().reset_index()
        temp_trend = temp_trend.merge(output, on="date", how="left")
        # it's ugly but we'll exploit the common column names
        # and then rename the underscore containing column names
        for kw in kw_list:
            norm_factor = np.ma.masked_invalid(
                temp_trend[kw+'_y']/temp_trend[kw+'_x']).mean()
            temp_trend[kw] = temp_trend[kw+'_x'] * norm_factor
        temp_trend = temp_trend[temp_trend.isnull().any(axis=1)]
        temp_trend['isPartial'] = temp_trend['isPartial_x']
        output = pd.concat(
            [output, temp_trend[['date', 'isPartial'] + kw_list]], axis=0, sort=False)

    # reorder columns in alphabetical order
    output = output[['date', 'isPartial']+kw_list]

    if not isPartial_col:
        output = output.drop('isPartial', axis=1)
    output = output[output['date'] >= from_date]
    if scale_cols:
        # the values in each column are relative to other columns
        # so we need to get the maximum value across the search columns
        max_val = float(output[kw_list].values.max())
        for col in kw_list:
            output[col] = 100.0*output[col]/max_val
    output = output.sort_values('date', ascending=False).reset_index(drop=True)
    return output


def get_recent_data(look_back=10):
    """
    Retrieves recent price and trend data for prediction in API

    Arguments
    look_back: int, number of previous days to predict on

    Returns:
    2 pandas DataFrames
    """

    today = date.today()
    from_date = (today - timedelta(days=(look_back + 3))).strftime("%Y-%m-%d")
    btc_price = get_btc_price(from_date)
    btc_trend = get_btc_trend(from_date)

    return btc_price, btc_trend


if __name__ == '__main__':
    from_date = '2014-01-01'
    print('Retrieving data starting from {}.'.format(from_date))

    output_dir = join(get_project_path(), 'data', 'raw')
    makedirs(output_dir, exist_ok=True)

    get_btc_price(from_date).to_csv(join(output_dir, 'btc_price.csv'))
    get_btc_trend(from_date).to_csv(join(output_dir, 'btc_trend.csv'))
