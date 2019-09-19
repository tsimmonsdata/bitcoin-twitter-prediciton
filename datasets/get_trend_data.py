from pytrendsdaily import getDailyData

trend_data = getDailyData('Bitcoin', 2017, 2019)
trend_data.to_json('btc_trends.json')
