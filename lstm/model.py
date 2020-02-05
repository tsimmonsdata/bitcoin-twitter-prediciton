from os.path import dirname, abspath, join
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, Flatten
from keras.layers import LSTM
from keras import Model
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

from lstm.transform_data import train_test_split
from lstm.helper import get_project_path


# Should convert this to a class so unit tests
# can be done for the training of the model

# import data
btc_df = read_csv(join(get_project_path(), 'data',
                       'processed', 'price_trend_data.csv'))
btc_df = btc_df.drop('date', axis=1)
X_train, y_train, X_test, y_test = train_test_split(btc_df)

# scale on training data
scale_x = StandardScaler()
X_train = scale_x.fit_transform(X_train)
X_test = scale_x.transform(X_test)
scale_y = StandardScaler()
y_train = scale_y.fit_transform(y_train.reshape(-1, 1))
y_test = scale_y.transform(y_test.reshape(-1, 1))

# reshape input to be 3D [samples, timesteps, features] for LSTM input
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM Model
dropout = 0.7
nodes = 16
epochs = 20
batch_size = 10
model = Sequential()
model.add(LSTM(nodes, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(dropout))
model.add(LSTM(nodes))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, verbose=0, shuffle=False,
          batch_size=batch_size, validation_data=(X_test, y_test))

# save model and scalers
model.save('lstm_model.h5')
joblib.dump(scale_x, 'scale_x.pkl')
joblib.dump(scale_y, 'scale_y.pkl')
