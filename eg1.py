"""
    Simple one-day forecast
"""

import data
import numpy as np

from sklearn.model_selection import train_test_split



sym = 'AAPL'

quotes = data.get_from_files(sym)

c = np.array(quotes.loc[:, 'close'], dtype = np.float32)
o = np.array(quotes.loc[:, 'open'], dtype = np.float32)
X = np.concatenate([o, c], axis=1)
y = (c >= o).astype(np.float32)

# Pre-condition the data: remove the mean of the day
#X = (X.T - 0.5 * (X[:, 0] + X[:, 1])).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Build a simple model

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, kernel_initializer='uniform'),
    tf.keras.layers.Dense(10, kernel_initializer='uniform'),
    tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

#import keras
#model = keras.Sequential([
#	keras.layers.Dense(2, kernel_initializer='uniform'),
#	keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid')
#])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, epochs=100)
