#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 11/12/18 5:23 PM
# @Author: Yongbo Wang
# @Email : yongbowin@outlook.com [OR] yonwang@redhat.com
# @File  : 06_logistic_regression.py
# @Desc  :

import tensorflow as tf

# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# set graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# set model weight and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# minimize error by cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init variable
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # run initializer
    sess.run(init)

    # training process
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # loop all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # calculate the cost and run optimizer
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

            # calculate avg loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
