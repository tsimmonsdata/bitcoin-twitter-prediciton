from pytrendsdaily import getDailyData

trend_data = getDailyData('Bitcoin', 2015, 2019)
trend_data.to_json('btc_trends.json')
