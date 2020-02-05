from datetime import date, timedelta
from pandas import DataFrame, concat, to_datetime, read_csv
from numpy import log, array
from os.path import join

from lstm.helper import get_project_path
from lstm.make_data import get_btc_price, get_btc_trend


def transform(price_df, trend_df):
    """
    Merges price and trend data and computes the logarithmic change of both

    Arguments:
    price_df: DataFrame or Numpy array
    trend_df: DataFrame or Numpy array

    Returns:
    pandas DataFrame
    """

    btc_df = price_df.merge(trend_df, on='date', how='inner')
    btc_df = btc_df.set_index('date')
    btc_df.index = to_datetime(btc_df.index)
    btc_df = btc_df.reindex(index=btc_df.index[::-1])
    # select only relevant columns, 'bitcoin' is the default name for the trend column
    btc_df = btc_df[['close', 'bitcoin']]

    # take the log difference
    btc_df['close'] = log(btc_df['close']) - \
        log(btc_df['close'].shift(1))
    btc_df['bitcoin'] = log(btc_df['bitcoin']) - \
        log(btc_df['bitcoin'].shift(1))
    btc_df.columns = ['log_returns', 'log_trend_chg']
    btc_df = btc_df[1:]

    return btc_df


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Convert time series DataFrame into a supervised learning structure
    with n_in timesteps per sample

    Arguments:
    data: DataFrame or Numpy array
    n_in: int, number of desired features
    n_out: int, number of desired targets
    dropnan: Boolean

    Returns:
    pandas DataFrame
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # keep var2(t) when using a single sample to predict var1(t+1)
    if agg.shape[0] != 1:
        agg.drop('var2(t)', axis=1, inplace=True)

    return agg


def train_test_split(data_transformed, split=0.7):
    """
    Split the transformed supervised learning data in train and test sets

    Arguments:
    data_transformed: DataFrame, processed by series_to_supervised
    split: float, train/test split percentage

    Returns:
    4 numpy Arrays
    """

    # split into train and test sets
    train_size = int(len(data_transformed) * split)
    train, test = data_transformed.values[:train_size], data_transformed.values[train_size:len(
        data_transformed)]

    # split in X and y
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    look_back = 10
    # read data
    input_path = join(get_project_path(), 'data', 'raw')
    price_df = read_csv(join(input_path, 'btc_price.csv'))
    trend_df = read_csv(join(input_path, 'btc_trend.csv'))
    # transform data
    btc_df = transform(price_df, trend_df)
    btc_df = series_to_supervised(btc_df, n_in=look_back)
    # save data
    btc_df.to_csv(join(get_project_path(), 'data',
                       'processed', 'price_trend_data.csv'))
