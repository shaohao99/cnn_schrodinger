# import tensorflow
import numpy as np
import tensorflow as tf
#import scipy as sp
np.set_printoptions(threshold=np.nan)

# user defined functions 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  #stddev=0.1, initialized as ~ N(0, 0.1^2)
  return tf.Variable(initial)  # return a Variable type, will be used in tf.global_variables_initializer()

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)  # initialized as constants: 0.1
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides=[batch, height, width, channels]

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ====== start main program ======
# Read potentials and energies from files
#pot_file='pot_eng2/he2d_potentials.csv'
#epsilon_file='pot_eng2/he2d_energies.csv'
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
ind_training=(1,)  # a tuple
ind_test=(0,)
for i in range(2,n_samples):
   if i%rtest == 0:
      ind_test = ind_test + (i,)   # index of test set: add item to a tuple
   else:
      ind_training = ind_training + (i,)  # index of training set

print('n_grids', 'nz', 'n_train', 'n_test', 'n_out')
print("%g %g %g %g %g"%(n_grids, n, n_train, n_test, n_out ) )

n_col = n_out + n_grids
train_data = np.zeros((n_train, n_col))
test_data = np.zeros((n_test, n_col))

eps = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_training), max_rows=1 )  # read the first row
train_data[:,0] = np.array([eps]).T[:,0]  # convert a 1d array (eps) to 2d (1 row, n_pot columns) array, then transpose it
train_data[:,1:n_col] = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_training) ) ) # read file

eps0 = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_test), max_rows=1 )  # read the first row
test_data[:,0] = np.array([eps0]).T[:,0]
test_data[:,1:n_col] = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_test) ) ) 
np.savetxt("true_eps.csv", test_data[:,0:1]) # save true epsilon in one column

#pot = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_training) ) ) # training data
#eps = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_training), max_rows=1 )  # read the first row
#pot0 = np.matrix.transpose( np.genfromtxt(pot_file, delimiter=',', usecols=(ind_test) ) ) # test data
#eps0 = np.genfromtxt(epsilon_file, delimiter=',', usecols=(ind_test), max_rows=1 )

#n_grids = pot.shape[1]  # number of cols in pot: number of 2D grids
#n = int(np.sqrt(n_grids))  # number of grids in 1D: should be multiple of both 5 and 4 (i.e. 20)
#n_pot = pot.shape[0]  # number of rows in pot: number of potentials to be used.
#n_epsilon = 1 #epsilon.shape[1] #[0]  # number of cols in epsilon: number of output energies
#n_pot0 = pot0.shape[0]  # number of rows in test potentials
#print('n_grids', 'nz', 'n_pot', 'n_pot0', 'n_epsilon')
#print("%g %g %g %g %g"%(n_grids, n, n_pot, n_pot0, n_epsilon ) )

#epsilon = np.zeros((n_pot, n_epsilon))
#epsilon = np.array([eps]).T  # convert a 1d array (eps) to 2d (1 row, n_pot columns) array, then transpose it. 
#epsilon0 = np.zeros((n_pot, n_epsilon))
#epsilon0 = np.array([eps0]).T

# Declare placeholder to hold input data
x = tf.placeholder(tf.float32, shape=[None, n_grids])  # shape=[None, n_grids] ?
y_ = tf.placeholder(tf.float32, shape=[None, n_out])  # shape=[None, n_epsilon] ?

# build cnn
fsize = 3
nchanel1 = 32 #32
nchanel2 = 16 #64
nchanel3 = 16 
print nchanel1, nchanel2, nchanel3

# first layer: cnn. n1*n1 x_image --> n1/2 * n1/2 * 32 h_pool1
# input data
x_image = tf.reshape(x, [-1, n, n, 1])  # reshape 1D tensor to 2D: [batch, in_height, in_width, in_channels]
  # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. The size of batch will be determined later.

W_conv1 = weight_variable([fsize, fsize, 1, nchanel1])  # [filter_height, filter_width, in_channels, out_channels]
b_conv1 = bias_variable([nchanel1])  # give a bias to each out channel

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # convolustion --> bias ---> relu
h_pool1 = max_pool_2x2(h_conv1)   # 2x2 max pooling: pick up the max value of the 2x2 block

# second layer: cnn. n1/2 * n1/2 *32 h_pool1 --> n1/4 * n1/4 *64 h_pool2
W_conv2 = weight_variable([fsize, fsize, nchanel1, nchanel2]) 
b_conv2 = bias_variable([nchanel2])

#h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# third layer: cnn. n1/4 * n1/4 *32 h_pool2 --> n1/8 * n1/8 *64 h_pool3
W_conv3 = weight_variable([fsize, fsize, nchanel2, nchanel3]) 
b_conv3 = bias_variable([nchanel3])

#h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# second last layer: fully connected. n1/4 * n1/4 * 64 h_pool2 -->  1024 h_fc1
n_pool = 3
pool_size = 2
fac_reduce = pool_size ** n_pool
n_full = n / fac_reduce # n #n/4

W_fc1 = weight_variable([n_full * n_full * nchanel3, 1024])
b_fc1 = bias_variable([1024])

#h_conv2_flat = tf.reshape(h_conv2, [-1, n_full*n_full*64])  #reshape 2D tensor to 1D. The -1 is for dimension of batch.
#h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)  # W*x+b --> relu
h_flat = tf.reshape(h_pool3, [-1, n_full*n_full*nchanel3]) #reshape 2D tensor to 1D. The -1 is for dimension of batch.
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)  # W*x+b --> relu

keep_prob = tf.placeholder(tf.float32)  # a scalar tensor, valued in [0,1]. The probability that each element is kept. It should be the same type as h_fc1 --- a placeholder here.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # Regularization: dropout to reduce overfitting, bring in nonlinearity. Randomly drop out elemenent with the probability 1 - keep_prob. Totally N*(1 - keep_prob) elements are dropped out.

# last layer: fully connected layer for final output, 1024 h_fc1_drop --> 10 y_conv
W_fc2 = weight_variable([1024, n_out])
b_fc2 = bias_variable([n_out])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   # Final output: y=W*x+b
#ntest1 = tf.size(y_conv)
#print("size of y_conv %d"%( ntest1 ) )

# Define loss function by cross entropy: between input y_ and output y_conv.
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# The function softmax_cross_entropy_with_logits is numerically stable. 
# It does: tf.nn.softmax( -tf.reduce_sum( y_ * tf.log(y_conv), 1) )

# Compute mean squared error for regression. This is used in Mills paper.
#mean_squared_error = tf.reduce_mean( tf.square(tf.subtract( y_, y_conv)), 1 )  # mean of 2nd col
mean_squared_error = tf.reduce_mean( tf.square(tf.subtract( y_, y_conv)) )  # mean of 2nd col

# Choose ADAM optimizer: returns W and b that niminize MSE. Also returns predicted y_conv?
lrate = 0.001  # 0.001
train_step = tf.train.AdadeltaOptimizer(lrate).minimize( mean_squared_error )
# tf.train.AdadeltaOptimizer (learning_rate=0.001) is used in Mills paper.
# tf.train.AdamOptimizer(1e-4).minimize(cross_entropy). learning_rate=1e-4. Uses moving averages of the parameters (momentum).
# Other options: train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) with a learning rate of 0.5. See https://www.tensorflow.org/versions/r1.1/api_guides/python/train#Optimizers .

# Declare an interactive session
sess = tf.InteractiveSession()
# initialize regression parameters: W=N(0,0.1^2), b=0.1 here.
sess.run(tf.global_variables_initializer())

# Really run training and evaluation
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
k = 0
iter = 0

for j in range(n_epoch):
  # random shuffle training data
  # Ref: https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks  
  np.random.shuffle(train_data)

  for i in range(n_batch):    # run batches through one epoch
    # Get data for the i-th batch
    istart = i*bsize #i*bsize/n_epoch #i*bsize
    iend = istart + bsize  # will use iend-1 in the following
    pot_batch = train_data[istart:iend, 1:n_col]
    eps_batch = train_data[istart:iend, 0:1]  # use 0:1 to get 2D (-column array), if use 0, it is a 1-d array
    #eps_batch = np.array([eps_batch]).T  # convert to 2D (1-column) array
    #print eps_batch

    if (i+1)%interv == 0:
       err_train[k,0] = iter
       err_train[k,1] = mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})
       #print("validation error %g"%mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0}))
       # t.eval() is a shortcut for calling tf.get_default_session().run(t), where t is a tensor. Here: Use the optimized W and b obtained from the (i-1)-th batch and the input x in the i-th batch to predict y_conv, then compute the MSE between y_conv end the true value y_. Keep all values of h_fc1 for evaluation.
       # train_error = mean_squared_error.eval(feed_dict={x: pot[...,i], y_: epsilon[..., i], keep_prob: 1.0}) 

    # Run traning: feed input data in the i-th batch, obtain optimized W and b
    train_step.run(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})  # dropout 50% of h_fc1 for training. Use feed_dict if Python code provides the ipnout data. 
    #train_step.run(feed_dict={x: pot, y_: epsilon, keep_prob: 1.0}) 
    if (i+1)%interv == 0:
       err_train[k,2] = mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0})
       print("training error: %g"%err_train[k,2])
       k = k + 1
       #print("training error %g"%mean_squared_error.eval(feed_dict={x: pot_batch, y_: eps_batch, keep_prob: 1.0}))
    iter = iter + 1  # count iterations

np.savetxt("training_error.csv", err_train, delimiter=",")

# Evaluate with test data: use the optimized W and b to predict y_conv in test sampels, and returns the MSE between y_conv and the trre value y_.
n_batch0 = n_test / bsize
pot_batch0 = np.zeros((bsize, n_grids))
eps_batch0 = np.zeros((bsize, 1))
err_test = np.zeros(n_batch0)
for i in range(n_batch0):
   istart0 = i*bsize
   iend0 = istart0 + bsize  # will use iend-1 in the following
   pot_batch0 = train_data[istart0:iend0, 1:n_col]
   eps_batch0 = train_data[istart0:iend0, 0:1]  # use 0:1 to get 2D (-column array), if use 0, it is a 1-d array
   err_test[i] = mean_squared_error.eval(feed_dict={x: pot_batch0, y_: eps_batch0, keep_prob: 1.0})
   print("test error of a batch: %g"% err_test[i] )

print("average test error: %g"% np.mean(err_test) )
print("test error of all test dtat: %g"%mean_squared_error.eval(feed_dict={x: test_data[:,1:n_col], y_: test_data[:,0:1], keep_prob: 1.0}))


#print "Predicted epsilon"
#print y_conv.eval(feed_dict={x: test_data[:,1:n_col], y_: test_data[:,0:1], keep_prob: 1.0})

y_pred = y_conv.eval(feed_dict={x: test_data[:,1:n_col], y_: test_data[:,0:1], keep_prob: 1.0})
np.savetxt("predicted_eps.csv", y_pred, delimiter=",")

# Notes: This program automatically runs on N(N>=) GPU on one node, if N GPUs are requested by SGE batch job scheduler.

