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
mask = y_close[0] < y_open[0]
ups = np.zeros((len(data), 2), dtype=np.float32)
ups[mask, 0] = 1.0     # Row 0 for down
ups[~mask, 1] = 1.0    # Row 1 for up

N_fc = 4     # Fully connected features
N_in = 2
kernel_size = 1
batch_size = 50

def nn(x):
    """
        Neural network for analyzing the time series
    """
    # N_conv1 = 16  # Filter features
    # N_conv2 = 32

    with tf.name_scope('reshape'):
        #x_rect = tf.reshape(x, [-1, kernel_size, N_in, 1])
        x_rect = tf.reshape(x, [-1, kernel_size * N_in])

    # with tf.name_scope('conv1'):
    #     W_conv1 = weight_variable([1, 1, 1, N_conv1])
    #     b_conv1 = bias_variable([N_conv1])
    #     h_conv1 = tf.nn.relu(tf.nn.conv2d(x_rect, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)

    # with tf.name_scope('pool1'):
    #     h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

    # with tf.name_scope('conv2'):
    #     W_conv2 = weight_variable([1, 1, N_conv1, N_conv2])
    #     b_conv2 = bias_variable([N_conv2])
    #     h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv2)

    # with tf.name_scope('pool2'):
    #     h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

    # with tf.name_scope('fc1'):
    #     # Down-sample by x2 and another x2, reshape both rows into one
    #     W_fc1 = weight_variable([int(kernel_size / 4) * N_in * N_conv2, N_fc])
    #     b_fc1 = bias_variable([N_fc])

    #     h_pool2_flat = tf.reshape(h_pool2, [-1, int(kernel_size / 4) * N_in * N_conv2])
    #     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([kernel_size * N_in, N_fc], name = 'w')
        b_fc1 = bias_variable([N_fc], name = 'b')
        h_fc1 = tf.matmul(x_rect, W_fc1) + b_fc1

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([N_fc, N_fc])
        b_fc2 = bias_variable([N_fc])
        h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
        # h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    with tf.name_scope('droptout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc_last'):
        W_fc = weight_variable([N_fc, 2])
        b_fc = bias_variable([2])

        y_conv = tf.nn.tanh(tf.matmul(h_fc_drop, W_fc) + b_fc)
        # y_conv = tf.matmul(h_fc_drop, W_fc) + b_fc
    return y_conv, keep_prob

def weight_variable(shape, name = 'Variable'):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name = 'Variable'):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

print('Setting up \033[38;5;214mTensorflow\033[0m ...')

x = tf.placeholder(tf.float32, [None, kernel_size, 2], name = 'x')
y_true = tf.placeholder(tf.float32, [None, 2], name = 'y_true')
y_conv, keep_prob = nn(x)

with tf.name_scope('loss'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_conv)

cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1.0e-3).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1.0e-3).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)

accuracy = tf.reduce_mean(correct_prediction)

# Initialize a session
sess = tf.Session()

tf.summary.scalar('accuracy', accuracy)

# Saving the graph
graph_location = './eg2'
merged_summary = tf.summary.merge_all()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

xx = np.zeros((batch_size, kernel_size, 2), dtype=np.float32)
yy = np.zeros((batch_size, 2), dtype=np.float32)

# i = 0
# for k in range(batch_size):
#     s = i + k
#     e = s + kernel_size
#     # xx[k, :, 0] = y_close[:, s : e]
#     # xx[k, :, 1] = -y_open[:, s : e]
#     xx[k, :, 0] = y_close[:, s : e]
#     xx[k, :, 1] =  - y_open[:, s : e]
#     yy[k, :] = ups[e - 1 : e, :]

run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)

b_fc1 = [v for v in tf.trainable_variables() if v.name == 'fc1/b:0']
w_fc1 = [v for v in tf.trainable_variables() if v.name == 'fc1/w:0']

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

z = 0
for i in range(0, len(data) - kernel_size - batch_size):
    for rep in range(10):
        for k in range(batch_size):
            s = i + k
            e = s + kernel_size
            xx[k, :, 0] = y_open[:, s : e]
            xx[k, :, 1] = y_close[:, s : e]
            yy[k, :] = ups[e - 1 : e, :]
        
        if z % 200 == 0:
            summ, train_accuracy, y_out = sess.run([merged_summary, accuracy, y_conv], 
                feed_dict = {x: xx, y_true: yy, keep_prob: 1.0},
                run_metadata = run_metadata,
                options = run_options
                )
            train_writer.add_run_metadata(run_metadata, 'step%03d' % z)
            train_writer.add_summary(summ, z)
            print('step %s/%s, training accuracy \033[38;5;120m%.3f\033[0m' % (rep, z, train_accuracy))
            # y_ind = sess.run(tf.argmax(y_out, 1))
            # ww, bb = sess.run([tf.reshape(w_fc1, [2, 4]), tf.reshape(b_fc1, [4])])
            # fc1v = sess.run(tf.matmul(tf.reshape(xx, [-1, 2]), ww) + bb)
            # for m in range(batch_size):
            #     print('   OC: %s -> w: %s; %s, b: %s -> FC1: %s -> FC3: %s -> %s -> %2d' % (xx[m, -1], ww[0], ww[1], bb, fc1v[m], yy[m, :], y_out[m], y_ind[m]))
        
        sess.run(train_step, feed_dict = {x: xx, y_true: yy, keep_prob: 0.5})
        z = z + 1

# Test
N = len(data) - i - kernel_size - 1
xx = np.zeros((N, kernel_size, 2), dtype = np.float32)
yy = np.zeros((N, 2), dtype = np.float32)
for k in range(N):
    s = i + k
    e = s + kernel_size
    xx[k, :, 0] = y_open[:, s : e]
    xx[k, :, 1] = y_close[:, s : e]
    yy[k, :] = ups[e - 1 : e, :]
accuracy, y_out = sess.run([accuracy, y_conv], feed_dict = {x: xx, y_true: yy, keep_prob: 1.0})
y_ind = sess.run(tf.argmax(y_out, 1))
print('Test, accuracy \033[38;5;120m%.3f\033[0m' % (accuracy))

ww, bb = sess.run([tf.reshape(w_fc1, [kernel_size * N_in, N_fc]), tf.reshape(b_fc1, [N_fc])])
fc1v = sess.run(tf.matmul(tf.reshape(xx, [-1, kernel_size * N_in]), ww) + bb)
for m in range(N):
    print('   OC: %s -> w: %s; %s, b: %s -> FC1: %s -> FC3: %s -> %s -> %2d' % (xx[m, -1], ww[0], ww[1], bb, fc1v[m], yy[m, :], y_out[m], y_ind[m]))
