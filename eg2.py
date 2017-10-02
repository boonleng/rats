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
y_close = np.array([data.loc[:, 'Close'].tolist()], dtype = np.float32)
y_open = np.array([data.loc[:, 'Open'].tolist()], dtype = np.float32)
mask = y_close[0] > y_open[0]
ups = np.zeros((len(data), 2), dtype=np.float32)
ups[~mask, 0] = 1.0   # Row 0 for down
ups[mask, 1] = 1.0    # Row 1 for up

kernel_size = 26

def nn(x):
    """
        Neural network for analyzing the time series
    """
    with tf.name_scope('reshape'):
        x_rect = tf.reshape(x, [-1, kernel_size, 1, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 1, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_rect, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

    with tf.name_scope('fc'):
        W_fc = weight_variable([int(kernel_size / 2) * 16, 512])
        b_fc = bias_variable([512])

        h_pool1_flat = tf.reshape(h_pool1, [-1, int(kernel_size  / 2) * 16])
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

x = tf.placeholder(tf.float32, [None, kernel_size])
y_true = tf.placeholder(tf.float32, [None, 2])
y_conv, keep_prob = nn(x)

with tf.name_scope('loss'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_conv)

cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

# Saving the graph
graph_location = './eg2'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

N = 5

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   for i in range(len(data) - kernel_size):
       train_accuracy = accuracy.eval(feed_dict={
                                      x: y_close[:, i : i + kernel_size],
                                      y_true: ups[i : i + kernel_size, :],
                                      keep_prob: 0.99
                                      })
       print('step %d, training accuracy %.3f' % (i, train_accuracy))
