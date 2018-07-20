import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)

# user defined functions 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  # W is initialized as ~ N(0, 0.1^2)
  return tf.Variable(initial)  

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)  # b is initialized as constants: 0.1
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') 

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ====== start main program ======
# read data
pot_file='../pot_eng7/he2d_potentials.csv'
epsilon_file='../pot_eng7/he2d_energies.csv'

n = 40
n_grids = n ** 2
n_out = 1
na = 200
nb = 200
n_samples = na*nb
rtest = 10
n_test = n_samples / rtest
n_train = n_samples - n_test
ind_training=(1,)  
ind_test=(0,)
for i in range(2,n_samples):
   if i%rtest == 0:
      ind_test = ind_test + (i,)   # index of test set
   else:
      ind_training = ind_training + (i,)  # index of training set

print('n_grids', 'nz', 'n_train', 'n_test', 'n_out')
print("%g %g %g %g %g"%(n_grids, n, n_train, n_test, n_out ) )

n_col = n_out + n_grids
train_data = np.zeros((n_train, n_col))
test_data = np.zeros((n_test, n_col))

eps = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_training), max_rows=1 )  # read the first row
train_data[:,0] = np.array([eps]).T[:,0]  
train_data[:,1:n_col] = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_training) ) ) 

eps0 = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_test), max_rows=1 )  # read the first row
test_data[:,0] = np.array([eps0]).T[:,0]
test_data[:,1:n_col] = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_test) ) ) 
np.savetxt("true_eps.csv", test_data[:,0:1]) # save true epsilon in one column

# Declare placeholder to hold input data
x = tf.placeholder(tf.float32, shape=[None, n_grids]) 
y_ = tf.placeholder(tf.float32, shape=[None, n_out]) 

# ======= build cnn ===========
fsize = 3
nchanel1 = 32 
nchanel2 = 16
nchanel3 = 16 
print nchanel1, nchanel2, nchanel3

# 1st layer: input data
x_image = tf.reshape(x, [-1, n, n, 1])  

# 2nd and 3rd layer: conv + pooling
W_conv1 = weight_variable([fsize, fsize, 1, nchanel1]) 
b_conv1 = bias_variable([nchanel1]) 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)   

# 4th and 5th layer: conv + pooling
W_conv2 = weight_variable([fsize, fsize, nchanel1, nchanel2]) 
b_conv2 = bias_variable([nchanel2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 6th and 7th layer: conv + pooling
W_conv3 = weight_variable([fsize, fsize, nchanel2, nchanel3]) 
b_conv3 = bias_variable([nchanel3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# 8th and 9th layer: flatten + fully connected.
n_pool = 3
pool_size = 2
fac_reduce = pool_size ** n_pool
n_full = n / fac_reduce 
W_fc1 = weight_variable([n_full * n_full * nchanel3, 1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(h_pool3, [-1, n_full*n_full*nchanel3]) 
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)  

keep_prob = tf.placeholder(tf.float32)  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # Optional: Drop to avoid overfitting

# last layer: fully connected  ---> output
W_fc2 = weight_variable([1024, n_out])
b_fc2 = bias_variable([n_out])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  

# Loss function = Training error
mean_squared_error = tf.reduce_mean( tf.square(tf.subtract( y_, y_conv)) ) 

# Use Adadelta optimizer to minimize loss function
lrate = 0.001  # learning rate
train_step = tf.train.AdadeltaOptimizer(lrate).minimize( mean_squared_error )

# Declare an interactive session and initialize W and b
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Divde training data into multiple batches
bsize =1000
n_batch = n_train / bsize
interv = 4
n_epoch = 500 #100
n_iter = n_batch * n_epoch
n_eval = n_iter / interv #+ n_epoch  # number of output error
print bsize, n_batch, n_epoch, n_iter
pot_batch = np.zeros((bsize, n_grids))
eps_batch = np.zeros((bsize, 1))
err_train = np.zeros((n_eval,3))

# Really run training and evaluation
k = 0
iter = 0
for j in range(n_epoch):    # loop of epochs
  # random shuffle training data
  np.random.shuffle(train_data)

  for i in range(n_batch):    # loop of batches in one epoch
    # Get data for the i-th batch
    istart = i*bsize 
    iend = istart + bsize  
    pot_batch = train_data[istart:iend, 1:n_col]
    eps_batch = train_data[istart:iend, 0:1]  

    if (i+1)%interv == 0:  # save valication error
       err_train[k,0] = iter
       err_train[k,1] = mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})

    # Run one tranining step
    train_step.run(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})  

    if (i+1)%interv == 0:  # save training error
       err_train[k,2] = mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})
       print("training error: %g"%err_train[k,2])
       k = k + 1
    iter = iter + 1  # count iterations

np.savetxt("training_error.csv", err_train, delimiter=",")

# Evaluate with test data
n_batch0 = n_test / bsize
pot_batch0 = np.zeros((bsize, n_grids))
eps_batch0 = np.zeros((bsize, 1))
err_test = np.zeros(n_batch0)
for i in range(n_batch0):
   istart0 = i*bsize
   iend0 = istart0 + bsize 
   pot_batch0 = train_data[istart0:iend0, 1:n_col]
   eps_batch0 = train_data[istart0:iend0, 0:1]  
   err_test[i] = mean_squared_error.eval(feed_dict={x: pot_batch0, y_: eps_batch0, keep_prob: 1.0})
   print("test error of a batch: %g"% err_test[i] )

print("average test error: %g"% np.mean(err_test) )
print("test error of all test data: %g"%mean_squared_error.eval(feed_dict={x: test_data[:,1:n_col], y_: test_data[:,0:1], keep_prob: 1.0}))

# Save predicted energies
y_pred = y_conv.eval(feed_dict={x: test_data[:,1:n_col], y_: test_data[:,0:1], keep_prob: 1.0})
np.savetxt("predicted_eps.csv", y_pred, delimiter=",")

