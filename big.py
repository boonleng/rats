import data
import matplotlib
import tensorflow as tf

quotes = data.get_old_data()
# quotes = data.get_old_data(reload = True)
sym = 'NVDA'
data = quotes[:, :, sym]
yy = data.loc[:, 'Close'].tolist()

# print(quotes)

print('Setting up \033[38;5;214mTensorflow\033[0m ...')

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.8], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_ = W * x + b

loss = tf.reduce_sum(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
# x_train = list(matplotlib.dates.date2num(data.index[0:-5].tolist()))
x_train = list(range(len(yy) - 5))
y_train = yy[0:-5]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('Training ...')
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

W_c, b_c, loss_c = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (W_c, b_c, loss_c))

# Create the model
# L = quotes.shape[2]
# x = tf.placeholder(tf.float32, [None, L], label = 'x')
# w = tf.Variable(tf.zeros([L, L]), label = 'w')
# b = tf.Variable(tf.zeros([L, L]), label = 'b')
# y = tf.matmul(x, W) + b

# y_ = tf.placeholder(tf.float32, [None, L])
