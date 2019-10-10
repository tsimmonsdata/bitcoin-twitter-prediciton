from cryptory import Cryptory
import pandas as pd

# initialise object
from_date = '2014-01-01'
my_cryptory = Cryptory(from_date=from_date)

# get historical bitcoin prices from coinmarketcap
price_data = my_cryptory.extract_coinmarketcap("bitcoin")

# get google trend data
trend_data = my_cryptory.get_google_trends(kw_list=["bitcoin"])

# save both to file
price_data.to_csv('btc_price.csv')
trend_data.to_csv('btc_trend.csv')
