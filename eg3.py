"""
    Simple one-day forecast
"""

import data
import tempfile
import matplotlib
import tensorflow as tf
import numpy as np
import pandas
import chart
import mystyle

quotes = data.get_old_data()

sym = 'AAPL'
data = quotes[:, :, sym]
y_close = np.array([data.loc[:, 'Close'].tolist()], dtype = np.float32)
y_open = np.array([data.loc[:, 'Open'].tolist()], dtype = np.float32)
mask = y_close[0] < y_open[0]
ups = np.zeros((len(data), 2), dtype=np.float32)
ups[mask, 0] = 1.0     # Row 0 for down
ups[~mask, 1] = 1.0    # Row 1 for up

N_fc = 2     # Fully connected features
N_in = 2
kernel_size = 1
batch_size = 50

def nn(x):
    """
        Neural network for analyzing the time series
    """
    with tf.name_scope('reshape'):
        x_rect = tf.reshape(x, [-1, kernel_size * N_in])

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([kernel_size * N_in, N_fc], name = 'w')
        b_fc1 = bias_variable([N_fc], name = 'b')
        h_fc1 = tf.matmul(x_rect, W_fc1) + b_fc1

    with tf.name_scope('droptout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc_last'):
        W_fc = weight_variable([N_fc, 2])
        b_fc = bias_variable([2])

        y_conv = tf.nn.tanh(tf.matmul(h_fc_drop, W_fc) + b_fc)
    return y_conv, keep_prob
