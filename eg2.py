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

sym = 'AAPL'
data = quotes[:, :, sym]
y_close = np.array([data.loc[:, 'Close'].tolist()], dtype = np.float32)
y_open = np.array([data.loc[:, 'Open'].tolist()], dtype = np.float32)
mask = y_close[0] > y_open[0]
ups = np.zeros((len(data), 2), dtype=np.float32)
ups[~mask, 0] = 1.0   # Row 0 for down
ups[mask, 1] = 1.0    # Row 1 for up

# ups[mask, 0] = 1.0    # Row 0 for down
# ups[~mask, 1] = 1.0    # Row 1 for up

kernel_size = 12

def nn(x):
    """
        Neural network for analyzing the time series
    """
    N_conv = 16 # Filter features
    N_fc = 128 # Fully connected features

    with tf.name_scope('reshape'):
        x_rect = tf.reshape(x, [-1, kernel_size, 2, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 1, 1, N_conv])
        b_conv1 = bias_variable([N_conv])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_rect, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([int(kernel_size / 2) * 2 * N_conv, N_fc])
        b_fc1 = bias_variable([N_fc])

        h_pool1_flat = tf.reshape(h_pool1, [-1, int(kernel_size  / 2) * 2 * N_conv])
        h_fc = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    with tf.name_scope('droptout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([N_fc, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.nn.relu(tf.matmul(h_fc_drop, W_fc2) + b_fc2)
    return y_conv, keep_prob

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev = 0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.001, shape = shape)
  return tf.Variable(initial)

print('Setting up \033[38;5;214mTensorflow\033[0m ...')

x = tf.placeholder(tf.float32, [None, kernel_size, 2])
y_true = tf.placeholder(tf.float32, [None, 2])
y_conv, keep_prob = nn(x)

with tf.name_scope('loss'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_conv)

cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

# Saving the graph
graph_location = './eg2'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

N = 10

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(0, len(data) - kernel_size - 5):
    if i == 0:
        xx = np.zeros((N, kernel_size, 2))
        yy = np.zeros((N, 2))
        for k in range(N):
            xx[k, :, 0] = y_open[:, i + k : i + k + kernel_size]
            xx[k, :, 1] = y_close[:, i + k : i + k + kernel_size]
            yy[k, :] = ups[i + k + kernel_size + 1 : i + k + kernel_size + 2, :]
    if i % 50 == 0:
        train_accuracy = accuracy.eval(session = sess,
                                       feed_dict = {x: xx, y_true: yy, keep_prob: 1.0})
        print('step %4d, training accuracy \033[38;5;120m%.3f\033[0m' % (i, train_accuracy))
        y_fore = y_conv.eval(session = sess, feed_dict = {x: xx, keep_prob: 1.0})
        y_ind = np.argmax(y_fore, 1)
        for m in range(len(y_fore)):
            print('   %6.2f %6.2f -> %2d / %s' % (xx[m, -1, 0], xx[m, -1, 1], y_ind[m], yy[m, :]))
    train_accuracy = accuracy.eval(session = sess,
                                   feed_dict = {x: xx, y_true: yy, keep_prob: 0.5})

# Test
N = len(data) - i - kernel_size - 1
xx = np.zeros((N, kernel_size, 2))
yy = np.zeros((N, 2))
for k in range(N):
    xx[k, :, 0] = y_open[:, i + k : i + k + kernel_size]
    xx[k, :, 1] = y_close[:, i + k : i + k + kernel_size]
    yy[k, :] = ups[i + k + kernel_size + 1 : i + k + kernel_size + 2, :]
y_fore = y_conv.eval(session = sess, feed_dict = {x: xx, keep_prob: 1.0})
print(y_fore)

y_fore = sess.run(tf.argmax(y_fore, 1))
print(y_fore)

# while i < len(data) - kernel_size:
#     up = tf.nn.softmax(y_fore)
#     print('Forecast %d : [%.3f, %.3f] -> %s  up/down: %s / %d' % (i, y_fore[0, 0], y_fore[0, 1], up == 1, y_fore[0, 0] > 0, ups[i, 0]))
#     i = i + 1
