"""
    Simple y = W * x + b model prediction
"""

import numpy as np
import tensorflow as tf

# Data
X = np.random.random((500, 2)).astype(np.float32)
w = np.array([1.0, 3.0], dtype=np.float32)
y = np.matmul(X, w)

# Add noise
y = y + 0.01 * np.random.random(y.shape).astype(np.float32)

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='linear', use_bias=False)
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.05),
              loss='mse', metrics=['mse'])
model.fit(X, y, batch_size=1, epochs=3)

# Retrieve the trained weights
m = model.layers[0].get_weights()
w_ = m[0].flatten()
if len(m) > 1:
    b_ = m[1].flatten()
else:
    b_ = None

print('Weights = [{:.4f}, {:.4f}] + {} vs {}'.format(w_[0], w_[1], b_, w))
