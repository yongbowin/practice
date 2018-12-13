# Import Tensorflow and Numpy
import tensorflow as tf
import numpy as np

# ======================
# Define the Graph
# ======================

# Create Placeholders For X And Y (for feeding in data)
X = tf.placeholder("float",[10, 10],name="X") # Our input is 10x10
Y = tf.placeholder("float", [10, 1],name="Y") # Our output is 10x1

# Create a Trainable Variable, "W", our weights for the linear transformation
initial_W = np.zeros((10,1))
W = tf.Variable(initial_W, name="W", dtype="float32")

# Define Your Loss Function
Loss = tf.pow(tf.add(Y,-tf.matmul(X,W)),2,name="Loss")

with tf.Session() as sess: # set up the session
    sess.run(tf.global_variables_initializer())
    # the first argument is the name of the tensorflow variabl you want to return
    # the second argument is the data for the placeholders
    Model_Loss = sess.run(
                Loss,
                {
                  X: np.random.rand(10,10),
                  Y: np.random.rand(10).reshape(-1,1)
                })
    print(Model_Loss)

    sess.close()