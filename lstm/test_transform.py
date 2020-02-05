import unittest
from os.path import join
import pandas as pd
import math
from numpy import log, random

from helper import get_project_path
from transform_data import transform, series_to_supervised, train_test_split


class TestTransformData(unittest.TestCase):
    def setUp(self):
        self.test_cols = ['log_returns', 'log_trend_chg']
        self.data_path_price = join(
            get_project_path(), 'data', 'raw', 'btc_price.csv')
        self.data_path_trend = join(
            get_project_path(), 'data', 'raw', 'btc_trend.csv')
        self.data_processed = join(
            get_project_path(), 'data', 'processed', 'price_trend_data.csv')
        self.price = pd.read_csv(self.data_path_price)
        self.trend = pd.read_csv(self.data_path_trend)
        self.processed = pd.read_csv(self.data_processed)

    def test_transform(self):
        transformed = transform(self.price, self.trend)
        self.price = self.price.reindex(index=self.price.index[::-1])
        log_return = log(self.price['close'].iloc[1]) - \
            log(self.price['close'].iloc[0])
        self.assertEqual(log_return, transformed['log_returns'].iloc[0])
        self.assertEqual(0, transformed.isna().sum().sum())

    def test_split(self):
        split = 0.7
        X_train, _, X_test, _ = train_test_split(
            self.processed, split)
        size = (self.processed).shape
        self.assertEqual(size[0], (X_train.shape[0] + X_test.shape[0]))
        self.assertEqual(size[1], (X_train.shape[1] + 1))

    def test_series_to_supervised(self):
        look_back = 10
        random.seed(42)
        series = random.rand(100, 2)
        supervised = series_to_supervised(series, look_back)
        feature_count = look_back * series.shape[1] + 1
        self.assertEqual(feature_count, supervised.shape[1])
        self.assertEqual(0, supervised.isna().sum().sum())
