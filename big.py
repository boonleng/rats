import data
# import tensorflow as tf

quotes = data.get_old_data()
# quotes = data.get_old_data(reload = True)

# Create the model
L = quotes.shape[2]
# x = tf.placeholder(tf.float32, [None, L])
# w = tf.Variable(tf.zeross[L, L])
# b = tf.Variable(tf.zeross[L, L])
# y = tf.matmul(x, W) + b

# y_ = tf.placeholder(tf.float32, [None, L])

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label = y_, logits = y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()

# for i in range(L):

# print('Saving to offline files ...')
# data.save_to_folder(quotes)

print(quotes)
