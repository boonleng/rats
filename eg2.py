"""
    Simple up/down prediction
"""

import data
import matplotlib
import tensorflow as tf
import numpy as np
import pandas
import chart
import mystyle

quotes = data.get_old_data()

sym = 'NVDA'
data = quotes[:, :, sym]
yy = np.array(data.loc[:, 'Close'].tolist(), dtype = np.float32)

def nn(x):
    """
        Neural network for analyzing the time series
    """
    with tf.name_scope('reshape'):
        x_rect = tf.reshape(x, [-1, 26, 1, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 1, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_rect, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

    with tf.name_scope('fc'):
        W_fc = weight_variable([13 * 16, 512])
        b_fc = bias_variable([512])

        h_pool1_flat = tf.reshape(h_pool1, [-1, 13 * 16])
        h_fc = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc) + b_fc)

    with tf.name_scope('droptout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([512, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

print('Setting up \033[38;5;214mTensorflow\033[0m ...')

x = tf.placeholder(tf.float32, [None, 26])
y_ = tf.placeholder(tf.float32, [None, 10])
y, keep_prob = nn(x)

#with tf.name_scope('loss'):
#	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
#    loss = tf.reduce_mean(tf.square(y_conv - y_))
#
#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

graph_location = './eg2'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
	tf.global_variables_initializer().run()

#sess = tf.Session()
#ss.run()

