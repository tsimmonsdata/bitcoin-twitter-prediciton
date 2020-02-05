# Btc_predict
Use a pre-trained LSTM Neural-Net with Bitcoin price data from coinmarketcap.com and Google trend data
through their pytrends library to predict the current day's closing price.

## Usage

Build the docker container

'''bash
./build 
'''

Run the docker container

'''bash
./run
'''

Obtain prediction at <http://localhost:5000/predict>
or <http://192.168.99.100:5000/predict> if using Docker Toolbox

## Futher Development

1. Implement Logging
2. Create template for the Flask app endpoint
3. Add a train endpoint to the Flask app
4. Find a better way to scale Google trend values
5. Create a Model class to offer model choices

