import data
import matplotlib
import tensorflow as tf
import numpy as np
import pandas
import chart
import mystyle

quotes = data.get_from_files()

sym = 'NVDA'
data = quotes[:, :, sym]
yy = np.array(data.loc[:, 'Close'].tolist(), dtype = np.float32)

# print(quotes)

print('Setting up \033[38;5;214mTensorflow\033[0m ...')

# Model parameters
W = tf.Variable([0.01], dtype = tf.float32)
b = tf.Variable([0.01], dtype = tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss
loss = tf.reduce_mean(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
N = 3

# Training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

for k in range(int(len(yy) / N)):
    x_train = np.multiply(np.arange(k * N, k * N + N, dtype = np.float32), 0.002)
    y_train = yy[k * N : k * N + N]
    sess.run(train, feed_dict = {x: x_train, y: y_train})
    if k % 10 == 0:
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("i: %5.2f   W: %7.4f   b: %7.4f   loss: %9.4f" % (x_train[0], curr_W, curr_b, curr_loss))

k = len(yy)
x0 = np.multiply(np.arange(k, k + N, dtype = np.float32), 0.002)
y0 = curr_W * x0 + curr_b
print('y0 = %s' % (y0))

# Gather the dates, properly indexed on weekedays only
dnum = matplotlib.dates.date2num(quotes.major_axis.tolist()[-1])
dates = quotes.major_axis.tolist()
k = 0
i = 0;
while k < N:
    day = matplotlib.dates.num2date(dnum + i).weekday()
    if day >= 0 and day < 5:
        dates.append(matplotlib.dates.num2date(dnum + i))
        k = k + 1
    i = i + 1

# Gather the results
qq = np.zeros((5, len(data) + N, 1))
qq[:, :, 0] = [
    data.loc[:, 'Open'].tolist() + list(np.add(y0, -3.0)),
    data.loc[:, 'High'].tolist() + list(np.add(y0, 1.0)),
    data.loc[:, 'Low'].tolist() + list(np.add(y0, -4.0)),
    data.loc[:, 'Close'].tolist() + list(y0),
    data.loc[:, 'Volume'].tolist() + np.multiply(np.ones(N), 10.0e6).tolist()
]

# Build a Pandas Panel for Chart
panel = pandas.Panel(data = qq, items = ['Open', 'High', 'Low', 'Close', 'Volume'], minor_axis = [sym], major_axis = dates)

# Make the Chart
#view = chart.Chart(90, color_scheme = 'sunset', forecast = N)
#view.set_xdata(dates[-90:])
#view.set_data(panel)
#view.savefig('figs/_test.png')

