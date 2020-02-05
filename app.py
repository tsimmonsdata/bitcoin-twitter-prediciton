from os.path import join
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import tensorflow as tf
from datetime import date

from lstm.make_data import get_recent_data
from lstm.transform_data import transform, series_to_supervised
from lstm.helper import get_project_path


app = Flask(__name__)

# load model and scalers
global graph
graph = tf.get_default_graph()
path = join(get_project_path(), 'lstm')
model = load_model(join(path, 'lstm_model.h5'))
scale_x = joblib.load(join(path, 'scale_x.pkl'))
scale_y = joblib.load(join(path, 'scale_y.pkl'))

# get recent price data to predict on
# There is a problem here with the cryptory package
# it only allows for trend data up to 2 days previous
# in addition to the need to inverse-normalize the
# trend data acquired from the get_recent_data call
look_back = 10
btc_price, btc_trend = get_recent_data()
# problems getting day previous trend data, easier to
# merge dataframes to keep dates consistent
price_data = btc_price.merge(btc_trend, on='date', how='inner')
date = str(date.today())

# transform for input into LSTM
data = transform(btc_price, btc_trend)
data = series_to_supervised(data[:10], n_in=(look_back - 1))
data = np.array(data).reshape(1, -1)
data = scale_x.transform(data)
data = data.reshape(1, 1, -1)


@app.route("/predict", methods=['GET'])
def predict():
    """
    Predict closing price given previous 10 days of price and trend data
    """

    with graph.as_default():
        prediction = model.predict(data)
    prediction = scale_y.inverse_transform(prediction)
    prediction = (price_data['close'].iloc[0] * np.exp(prediction))[0][0]

    return jsonify('Predicted closing price for {}: {:.2f}'.format(date, prediction))


if __name__ == '__main__':
    app.run(debug=True)
