"""
    Simple up-down logic
"""

import data
import numpy as np
from sklearn.model_selection import train_test_split

sym = 'NVDA'

quotes = data.get_from_files(sym)

c = np.array(quotes.loc[:, 'close'], dtype = np.float32)
o = np.array(quotes.loc[:, 'open'], dtype = np.float32)
X = np.concatenate([o, c], axis=1)
y = (c > o).astype(np.float32)

# Pre-condition the data
b = X[:, 0]
X = (X.T - b).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Build a simple model

#
# Using Keras directly
#

#import keras
#model = keras.Sequential([
#	keras.layers.Dense(2, kernel_initializer='uniform'),
#	keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid')
#])
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

#
# Using Tensorflow
#

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, kernel_initializer='uniform'),
    tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid')
])
model.compile(optimizer=tf.train.AdamOptimizer(0.003),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=100)
model.evaluate(X_test, y_test)

# Visualize
def test_grid(r):
    f, g = np.meshgrid(r, r)
    x = np.vstack((f.flatten(), g.flatten())).T
    return x

# Post processing
X = (X.T + b).T

r = np.arange(50, 310, 2)
XX = test_grid(r)
XX = (XX.T - XX[:, 0]).T
y_pred = model.predict([XX])
y_pred = y_pred.reshape((len(r), len(r)))

import matplotlib.pyplot as plt

extent = (r[0]-0.5, r[-1]-0.5, r[0]-0.5, r[-1]-0.5)

plt.figure(dpi=144)
plt.imshow(y_pred, extent=extent, cmap='Spectral', origin='lower')
plt.plot(X[:, 0], X[:, 1], 'wo', markersize=1)
plt.axis(extent)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Output')
#plt.savefig('blob/eg2.png')
